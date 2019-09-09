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

    # Reward Metrics
    MAX_WAITING_TIME = 30  # minutes
    MIN_IDLE_TIME = 15  # minutes

    def __init__(self):
        super().__init__(use_batsim=False)
        self.reservation_size = 0
        self.scheduler = SAFBackfilling()

        for ps in self.rjms.platform.nodes[0].power_states:
            if ps.type == PowerStateType.computation:
                p_idle = ps.power_min
            elif ps.type == PowerStateType.switching_off:
                p_turn_off = ps.power_min
                t_turn_off = 1 / ps.speed
            elif ps.type == PowerStateType.switching_on:
                p_turn_on = ps.power_min
                t_turn_on = 1 / ps.speed

        self.p_max = max(p_idle, p_turn_off, p_turn_on)
        p_idle /= self.p_max
        p_turn_off /= self.p_max
        p_turn_on /= self.p_max
        p_switch = (p_turn_off * t_turn_off) + (p_turn_on * t_turn_on)

        self.switch_penalty = ((self.MIN_IDLE_TIME + self.ACT_INTERVAL) * p_idle) / p_switch
        self.qos_penalty = (p_turn_on * t_turn_on) * self.switch_penalty

    def reset(self):
        self.reservation_size = 0
        self.last_platform_state = {
            n.id: n.state.type for n in self.rjms.platform.nodes}
        return super().reset()

    def step(self, action):
        assert self.rjms.is_running, "Simulation is not running."

        self._set_off_reservation_size(action)

        reserved = self.rjms.agenda.get_reserved_time(self.rjms.current_time)
        reserved.sort()
        jobs_to_start = self.scheduler.schedule(
            self.rjms.jobs_queue[:self.MAX_QUEUE_SZ],
            reserved[self.reservation_size * self.rjms.platform.nodes[0].nb_resources:])

        for job_id in jobs_to_start:
            self.rjms.allocate(job_id)

        self.rjms.start_ready_jobs()

        # This should occur before proceeding time because new jobs can be submitted.
        reward = self._get_reward()

        # self.history.append(self._get_obs())
        # while self.rjms.is_running and len(jobs_ready) == 0 and not any(n.is_switching_off or n.is_switching_on or n.is_idle for n in self.rjms.platform.nodes):
        #    next_event_time = int(self.rjms.simulator.get_next_event_time())
        #    last = max(self.rjms.current_time, next_event_time - self.HISTORY_LENGHT)
        #    for i in range(last+1, next_event_time, int(self.ACT_INTERVAL)):
        #        self.rjms.proceed_time(i)
        #        self.history.append(self._get_obs())
        #    self.rjms.proceed_time(self.rjms.current_time + self.ACT_INTERVAL)
        #    self.history.append(self._get_obs())
        #    jobs_ready = self.scheduler.schedule(
        #        self.rjms.jobs_queue[:self.MAX_QUEUE_SZ], reserved)

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
        if size <= len(nodes_available):
            self.reservation_size = size  # min(size, len(nodes_available))

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

    def _get_info(self):
        # def get_next_submissions():
        #    upper_bound = self.rjms.current_time + 360
        #    next_submission = next((idx for idx, j in enumerate(
        #        self.current_workload.jobs) if j.subtime > self.rjms.current_time), None)
        #    next_submissions = []
        #    if next_submission:
        #        for j in itertools.takewhile(lambda j: j.subtime <= upper_bound, self.current_workload.jobs[next_submission:]):
        #            next_submissions.append({
        #                'subtime': j.subtime,
        #                'res': j.res,
        #                'walltime': j.walltime,
        #                'expected_time_to_start': j.expected_time_to_start
        #            })
        #    return next_submissions
        #
        # if not self.rjms.is_running:
            # return {k: v for k, v in self.scheduler_monitor.info.items()}
        # else:
        return {}

    def _get_reward(self):
        energy_waste = 0
        for n in self.rjms.platform.nodes:
            p_max = max(ps.power_min for ps in n.power_states)
            if n.is_switching_on or n.is_switching_off :
                energy_waste += (n.power / p_max) * self.switch_penalty
            elif n.is_idle:
                energy_waste += n.power / p_max
        energy_waste /= self.rjms.platform.nb_nodes

        total_res = qos = 0
        for job in self.rjms.jobs_queue:
            res_norm = (job.res / self.rjms.platform.nb_resources)
            job_qos = (res_norm * self.qos_penalty) / self.MAX_WAITING_TIME
            qos += job_qos
            total_res += job.res
            if total_res >= self.rjms.platform.nb_resources * 3:
                break
        # if self.reservation_size != 0 and len(queue) > 0:
        #    r = self.rjms.agenda.get_reserved_time(self.rjms.current_time)
        #    old_p = self.scheduler.priority_job
        #    jobs_ready = self.scheduler.schedule(queue, r)
        #    self.scheduler.priority_job = old_p
        #    # for job_id in jobs_ready:
        #    #    job = next(j for j in queue if j.id == job_id)
        #    # if (self.rjms.current_time - job.subtime) >= self.MAX_WAITING_TIME:
        #    #    qos += job.res / self.rjms.platform.nb_resources
        #    qos = len(jobs_ready) / self.MAX_WAITING_TIME
        #res = sum(j.res for j in self.rjms.jobs_queue)#[:self.MAX_QUEUE_SZ]
        #nb_cores = self.rjms.platform.nodes[0].nb_resources
       # qos = min(self.rjms.platform.nb_nodes*self.MAX_WAITING_TIME, math.ceil(res / nb_cores))
        #qos /= self.rjms.platform.nb_nodes
        return -1 * (energy_waste + qos)

    def _get_obs(self, reward=0):
        obs = {}

        # Update scheduler
        reserved = self.rjms.agenda.get_reserved_time(self.rjms.current_time)
        reserved.sort()
        nb_reserved = self.reservation_size * \
            self.rjms.platform.nodes[0].nb_resources
        self.scheduler.schedule(
            self.rjms.jobs_queue[:self.MAX_QUEUE_SZ],
            reserved[nb_reserved:])

        obs['queue'] = np.asarray(
            [[j.subtime, j.res, j.walltime, j.expected_time_to_start]
                for j in self.rjms.jobs_queue]
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

        obs['free'] = len(
            self.rjms.agenda.get_available_resources()) - nb_reserved
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
            shape=(self.MAX_QUEUE_SZ, 4),
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
            'platform': platform,
            'agenda': agenda,
            'free': spaces.Discrete(self.rjms.platform.nb_nodes + 1),
            'reservation_size': spaces.Discrete(self.rjms.platform.nb_nodes + 1),
            'time': spaces.Box(low=0, high=np.iinfo(int).max, shape=(), dtype=np.int),
            'reward': spaces.Box(low=0, high=np.iinfo(int).max, shape=(), dtype=np.int)
        })
        act_space = spaces.Discrete(self.rjms.platform.nb_nodes + 1)
        self.rjms.close()
        return obs_space, act_space
