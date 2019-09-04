import math
import itertools
import os

import pandas as pd
import numpy as np
import gym
from gym import spaces

from gridgym.envs.grid_env import GridEnv
from gridgym.envs.simulator.utils.schedulers import EASYBackfilling
from gridgym.envs.simulator.resource import ResourceState, PowerStateType
from gridgym.envs.simulator.utils.monitors import *


class OffReservationEnv(GridEnv):
    MAX_QUEUE_SZ = 20
    ACT_INTERVAL = 1  # minutes
    SIMULATION_TIME = None  # 60 * 60 * 24  # minutes
    TRACE = True

    # Reward Metrics
    MAX_WAITING_TIME = 60 # minutes
    OFF_REWARD = 0.01  # minutes
    MIN_IDLE_TIME = 10  # minutes

    def __init__(self):
        super().__init__(use_batsim=False)
        self.reservation_size = 0
        self.scheduler = EASYBackfilling()
        self._last_node_state = {
            n.id: n.state.type for n in self.rjms.platform.nodes}

        for ps in self.rjms.platform.nodes[0].power_states:
            if ps.type == PowerStateType.computation:
                idle_total = ps.power_min
            elif ps.type == PowerStateType.switching_off:
                turnoff_total = ps.power_min
                turnoff_time = 1 / ps.speed
            elif ps.type == PowerStateType.switching_on:
                turnon_total = ps.power_min
                turnon_time = 1 / ps.speed

        max_power = max(idle_total, turnoff_total, turnon_total)
        idle_total /= max_power
        turnoff_total = (turnoff_total / max_power) * turnoff_time
        turnon_total = (turnon_total / max_power) * turnon_time
        t = (self.MIN_IDLE_TIME + self.ACT_INTERVAL)
        t_off = t - turnoff_time - turnon_time
        #self.switch_penalty = (t * idle_total) / (turnoff_total + turnon_total - (t_off * self.OFF_REWARD))
        self.switch_penalty = (t * idle_total - turnoff_total - turnon_total + t_off * self.OFF_REWARD) / 2.
        #self.qos_penalty = (turnon_total * self.switch_penalty)
        self.qos_penalty = (turnon_total + self.switch_penalty)

    def reset(self):
        self.reservation_size = 0
        return super().reset()

    def step(self, action):
        assert self.rjms.is_running, "Simulation is not running."

        self._set_off_reservation_size(action)

        reserved_time = self.rjms.agenda.get_reserved_time(
            self.rjms.current_time)
        reserved_time.sort()
        jobs_to_start = self.scheduler.schedule(
            self.rjms.jobs_queue[:self.MAX_QUEUE_SZ],
            reserved_time[self.reservation_size * self.rjms.platform.nodes[0].nb_resources:])

        for job_id in jobs_to_start:
            self.rjms.allocate(job_id)

        self.rjms.start_ready_jobs()

        # This should occur before proceeding time because new jobs can be submitted.
        reward = self._get_reward()

        self.rjms.proceed_time(
            self.rjms.current_time + self.ACT_INTERVAL)

        obs = self._get_obs()
        done = not self.rjms.is_running
        if done and self.TRACE:
            self.job_monitor.to_csv(self.OUTPUT + "_jobs.csv")
            self.scheduler_monitor.to_csv(self.OUTPUT + "_schedule.csv")
            self.res_monitor.to_csv(self.OUTPUT + "_machine_states.csv")
            self.pstate_monitor.to_csv(self.OUTPUT + "_pstate_changes.csv")
            self.energy_monitor.to_csv(self.OUTPUT + "_consumed_energy.csv")

        info = self._get_info()
        return obs, reward, done, info

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
        # self.rjms.resource_monitor.statistics
        #info['next_submissions'] = get_next_submissions()
        # if int(self.rjms.current_time) <= self.ACT_INTERVAL:
        #    info['nodes_profile'] = [
        #        [{'min': ps.minimum, 'max': ps.maximum, 'type': ps.type} for ps in n.power_states] for n in self.rjms.platform.nodes
        #    ]

    def _get_reward(self):
        energy_waste = qos = 0
        nb_off = nb_switches = 0

        for n in self.rjms.platform.nodes:
            max_power = max(ps.power_min for ps in n.power_states)
            if n.is_switching_on or n.is_switching_off:
                energy_waste += (n.power / max_power)
                if n.state.type != self._last_node_state[n.id]:
                    nb_switches += 1
                    self._last_node_state[n.id] = n.state.type
            elif n.is_idle:
                energy_waste += (n.power / max_power)

            #if n.is_switching_on or n.is_switching_off:
            #    energy_waste += (n.power / max_power) * self.switch_penalty
            #elif n.is_idle:
            #    energy_waste += n.power / max_power
            elif n.is_off:
                nb_off += 1

        nb_switches /= self.rjms.platform.nb_nodes
        nb_switches *= self.switch_penalty

        energy_waste /= self.rjms.platform.nb_nodes
        nb_off /= self.rjms.platform.nb_nodes
        nb_off *= self.OFF_REWARD

        queue = self.rjms.jobs_queue[:self.MAX_QUEUE_SZ]
        if self.reservation_size != 0 and len(queue) > 0:
            jobs = self.scheduler.schedule(
                queue,
                self.rjms.agenda.get_reserved_time(self.rjms.current_time))
            for job_id in jobs:
                job = next(j for j in queue if j.id == job_id)
                stretch = min(1, (self.rjms.current_time - job.subtime) / job.walltime)
                if stretch >= 0.5:
                    job_qos = ((job.res / self.rjms.platform.nb_resources) * self.qos_penalty) * stretch
                    qos += job_qos
        return nb_off + -1 * (energy_waste + qos + nb_switches)

    def _set_off_reservation_size(self, size):
        if size == 0:
            self.reservation_size = 0
            return

        # delimit the reservation size
        nodes_available = self.rjms.agenda.get_available_nodes()
        self.reservation_size = min(size, len(nodes_available))

        # Just update the scheduler to see if we delay the priority job
        reserved_time = np.sort(
            self.rjms.agenda.get_reserved_time(self.rjms.current_time))
        self.scheduler.schedule(
            self.rjms.jobs_queue[:self.MAX_QUEUE_SZ],
            reserved_time[self.reservation_size * self.rjms.platform.nodes[0].nb_resources:])

        pjob = self.scheduler.priority_job
        # free some resources for priority job
        if pjob and pjob.expected_time_to_start <= self.ACT_INTERVAL:
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

    def _get_obs(self):
        obs = {}
        # let's update the scheduler decision
        reserved_time = np.sort(
            self.rjms.agenda.get_reserved_time(self.rjms.current_time))
        self.scheduler.schedule(
            self.rjms.jobs_queue[:self.MAX_QUEUE_SZ],
            reserved_time[self.reservation_size * self.rjms.platform.nodes[0].nb_resources:])

        # obs['queue'] = np.full(
        #    fill_value=-1,
       #     shape=self.observation_space.spaces['queue'].shape,
       #     dtype=self.observation_space.spaces['queue'].dtype)
        # for i, j in enumerate(self.rjms.jobs_queue):
        #    obs['queue'][i] = np.asarray(
        #        [j.subtime, j.res, j.walltime, j.expected_time_to_start])

        obs['queue'] = np.asarray(
            [[j.subtime, j.res, j.walltime, j.expected_time_to_start] for j in self.rjms.jobs_queue])

        obs['platform'] = np.zeros(
            shape=self.observation_space.spaces['platform'].shape,
            dtype=self.observation_space.spaces['platform'].dtype)

        for n in self.rjms.platform.nodes:
            obs['platform'][int(n.id)] = np.asarray(
                [r.state.value for r in n.resources])

        obs['agenda'] = np.zeros(
            shape=self.observation_space.spaces['agenda'].shape,
            dtype=self.observation_space.spaces['agenda'].dtype)

        for i, p in enumerate(self.rjms.agenda.get_progress(self.rjms.current_time)):
            obs['agenda'][i] = p

        obs['run'] = np.zeros(
            shape=self.observation_space.spaces['run'].shape,
            dtype=self.observation_space.spaces['run'].dtype)

        for i, j in enumerate(self.rjms.jobs_running):
            runtime = self.rjms.current_time - j.start_time
            remaining = (1 - (runtime / j.walltime))
            obs['run'][i] = [j.res, remaining]

        obs['reservation_size'] = self.reservation_size
        obs['time'] = self.rjms.current_time
        return obs

    def _get_space(self):
        self.rjms.start(platform_fn=self.PLATFORM, output_dir=self.OUTPUT,
                        simulation_time=self.SIMULATION_TIME)

        queue = spaces.Box(
            low=-1,
            high=np.iinfo(int).max,
            shape=(self.MAX_QUEUE_SZ, 4),
            dtype=np.int)
        platform = spaces.Box(
            low=0,
            high=np.iinfo(int).max,
            shape=(self.rjms.platform.nb_nodes, max(
                n.nb_resources for n in self.rjms.platform.nodes)),
            dtype=np.int)
        agenda = spaces.Box(
            low=0,
            high=1,
            shape=(self.rjms.platform.nb_resources,),
            dtype=np.float)

        running = spaces.Box(
            low=0,
            high=1,
            shape=(self.rjms.platform.nb_resources, 2),
            dtype=np.float)

        obs_space = spaces.Dict({
            'queue': queue,
            'platform': platform,
            'agenda': agenda,
            'run': running,
            'reservation_size': spaces.Discrete(self.rjms.platform.nb_nodes + 1),
            'time': spaces.Box(low=0, high=np.iinfo(int).max, shape=(), dtype=np.int)
        })
        act_space = spaces.Discrete(self.rjms.platform.nb_nodes + 1)
        self.rjms.close()
        return obs_space, act_space
