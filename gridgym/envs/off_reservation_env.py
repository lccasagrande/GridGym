import math
import itertools
import os

import pandas as pd
import numpy as np
import gym
from gym import spaces

from gridgym.envs.grid_env import GridEnv
from gridgym.envs.simulator.utils.schedulers import EASYBackfilling, SAFBackfilling
from gridgym.envs.simulator.resource import ResourceState, PowerStateType
from gridgym.envs.simulator.utils.monitors import *


class OffReservationEnv(GridEnv):
    MAX_QUEUE_SZ = 10
    ACT_INTERVAL = 1  # minutes
    SIMULATION_TIME = 1440  # 60 * 60 * 24  # minutes
    TRACE = True

    # REWARD METRIC
    QOS_STRETCH = 0.5

    def __init__(self):
        super().__init__(use_batsim=False)
        self.reservation_size = 0
        self.scheduler = EASYBackfilling()
        self.workload_name = ""
        if self.TRACE:
            self.scheduler_monitor = SchedulerStatsMonitor(
                self.rjms.simulator, self.QOS_STRETCH)
            self.res_monitor = ResourceStatesEventMonitor(self.rjms.simulator)
            self.job_monitor = JobMonitor(self.rjms.simulator)
            self.pstate_monitor = ResourcePowerStatesEventMonitor(
                self.rjms.simulator)
            self.energy_monitor = EnergyEventMonitor(self.rjms.simulator)

    def reset(self):
        self.reservation_size = 0
        o = super().reset()
        self.workload_name = self.job_submitter.current_workload.name
        return o

    def _get_queue(self, maxlen=0):
        assert maxlen >= 0
        s_queue = []
        if self.rjms.queue_lenght > 0:
            queue = self.rjms.jobs_queue
            s_queue.append(queue[0])
            for j in sorted(queue[1:], key=lambda j: j.walltime * j.res):
                if len(s_queue) == maxlen:
                    break
                s_queue.append(j)
        return s_queue
        # return self.rjms.jobs_queue  # np.asarray(queue)

    def step(self, action):
        assert self.rjms.is_running, "Simulation is not running."

        self._set_off_reservation_size(action)

        reserved = self.rjms.agenda.get_reserved_time(self.rjms.current_time)
        reserved.sort()
        jobs_to_start = self.scheduler.schedule(
            self._get_queue(self.MAX_QUEUE_SZ),
            reserved[self.reservation_size * self.rjms.platform.nodes[0].nb_resources:])

        for job_id in jobs_to_start:
            self.rjms.allocate(job_id)

        self.rjms.start_ready_jobs()

        # This should occur before proceeding time because new jobs can be submitted.
        reward = self._get_reward()

        self.rjms.proceed_time(self.rjms.current_time + self.ACT_INTERVAL)
        if self.rjms.is_running:
            self.rjms.start_ready_jobs()

        obs = self._get_obs(reward)
        done = not self.rjms.is_running
        if done and self.TRACE:
            self.job_monitor.to_csv(self.OUTPUT + "_jobs.csv")
            self.scheduler_monitor.to_csv(self.OUTPUT + "_schedule.csv")
            self.res_monitor.to_csv(self.OUTPUT + "_machine_states.csv")
            self.pstate_monitor.to_csv(self.OUTPUT + "_pstate_changes.csv")
            self.energy_monitor.to_csv(self.OUTPUT + "_consumed_energy.csv")

        info = self._get_info()
        return obs, reward, done, info

    def _set_off_reservation_size(self, size):
        if size == 0:
            self.reservation_size = 0
            return

        # delimit the reservation size
        nodes_available = self.rjms.agenda.get_available_nodes()
        self.reservation_size = min(size, len(nodes_available))

        # Just update the scheduler to see if we delay the priority job
        pjob = self.rjms.jobs_queue[0] if self.rjms.queue_lenght > 0 else None
        # free some resources for priority job
        if pjob and pjob.expected_time_to_start == 0:
            nb_nodes_needed = math.ceil(
                pjob.res / self.rjms.platform.nodes[0].nb_resources)
            nb_nodes_free = self.rjms.platform.nb_nodes - nb_nodes_needed
            self.reservation_size = min(self.reservation_size, nb_nodes_free)

        # find how many nodes must be turned off
        nodes_on = [n.id for n in nodes_available if n.is_on]
        if nodes_on:
            nb_not_on = len(nodes_available) - len(nodes_on)
            nb_to_turn_off = max(0, self.reservation_size - nb_not_on)
            self.rjms.turn_off(*nodes_on[:nb_to_turn_off])

    def _get_reward(self):
        energy_waste = 0
        for n in self.rjms.platform.nodes:
            p_max = max(ps.power_min for ps in n.power_states)
            if n.is_switching_off or n.is_switching_on or n.is_idle:
                energy_waste += n.power / p_max
        energy_waste /= self.rjms.platform.nb_nodes

        queue = self._get_queue(self.MAX_QUEUE_SZ)
        qos = 0
        if self.reservation_size != 0 and len(queue) > 0:
            r = self.rjms.agenda.get_reserved_time(self.rjms.current_time)
            jobs_ready = self.scheduler.schedule(queue, r)
            for job_id in jobs_ready:
                job = next(j for j in queue if j.id == job_id)
                if (self.rjms.current_time - job.subtime) / job.walltime >= self.QOS_STRETCH:
                    qos += job.res
            qos /= self.rjms.platform.nb_resources
        return -1 * (energy_waste + qos)

    def _get_obs(self, reward=0):
        obs = {}

        # Update scheduler
        reserved = self.rjms.agenda.get_reserved_time(self.rjms.current_time)
        reserved.sort()
        nb_reserved = self.reservation_size * \
            self.rjms.platform.nodes[0].nb_resources
        self.scheduler.schedule(self._get_queue(
            self.MAX_QUEUE_SZ), reserved[nb_reserved:])

        obs['queue'] = np.asarray(
            [
                [j.subtime, j.res, j.walltime, j.expected_time_to_start, j.user, int(j.profile[:j.profile.rfind('.')])] for j in self._get_queue()
            ],
            dtype=self.observation_space.spaces['queue'].dtype
        )

        obs['jobs_running'] = np.asarray(
            [
                [j.start_time, j.res, j.walltime, j.user, int(j.profile[:j.profile.rfind('.')])] for j in self.rjms.jobs_running
            ]
        )

        obs['platform'] = np.zeros(
            shape=self.observation_space.spaces['platform'].shape,
            dtype=self.observation_space.spaces['platform'].dtype)

        for n in self.rjms.platform.nodes:
            obs['platform'][int(n.id)] = [r.state.value for r in n.resources]

        obs['agenda'] = np.zeros(
            shape=self.observation_space.spaces['agenda'].shape,
            dtype=self.observation_space.spaces['agenda'].dtype)

        for i, p in enumerate(self.rjms.agenda.get_progress(self.rjms.current_time)):
            obs['agenda'][i] = p

        obs['reservation_size'] = self.reservation_size
        obs['time'] = self.rjms.current_time
        obs['reward'] = reward
        return obs

    def _get_space(self):
        self.rjms.start(
            platform_fn=self.PLATFORM,
            output_dir=self.OUTPUT,
            simulation_time=self.SIMULATION_TIME)

        queue = spaces.Box(
            low=-1,
            high=np.iinfo(int).max,
            shape=(self.MAX_QUEUE_SZ, 6),
            dtype=np.int)

        jobs_running = spaces.Box(
            low=-1,
            high=np.iinfo(int).max,
            shape=(self.rjms.platform.nb_resources, 5),
            dtype=np.int)

        platform = spaces.Box(
            low=0,
            high=5,
            shape=(self.rjms.platform.nb_nodes,
                   self.rjms.platform.nodes[0].nb_resources),
            dtype=np.float)

        agenda = spaces.Box(
            low=0,
            high=1,
            shape=(self.rjms.platform.nb_resources,),
            dtype=np.float)

        obs_space = spaces.Dict({
            'queue': queue,
            'jobs_running': queue,
            'platform': platform,
            'agenda': agenda,
            'reservation_size': spaces.Discrete(self.rjms.platform.nb_nodes + 1),
            'time': spaces.Box(low=0, high=np.iinfo(int).max, shape=(), dtype=np.int),
            'reward': spaces.Box(low=0, high=np.iinfo(int).max, shape=(), dtype=np.int)
        })
        act_space = spaces.Discrete(self.rjms.platform.nb_nodes + 1)
        self.rjms.close()
        return obs_space, act_space

    def _get_info(self):
        info = {'workload_name': self.workload_name}
        return info
