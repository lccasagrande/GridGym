import math
import itertools
import os

import pandas as pd
import numpy as np
import gym
from gym import spaces


from gridgym.envs.grid_env import GridEnv
from batsim_py.rjms import RJMSHandler
from batsim_py.utils.schedulers import EASYBackfilling, SAFBackfilling
from batsim_py.resource import ResourceState, PowerStateType
from batsim_py.utils.monitors import *


class RJMSWrapper(RJMSHandler):
    def __init__(self, max_queue_sz=None, use_batsim=False):
        super().__init__(use_batsim=use_batsim)
        if max_queue_sz:
            assert max_queue_sz > 0

        self.nb_nodes = -1
        self.max_queue_sz = max_queue_sz

    @property
    def jobs_queue(self):
        queue = super().jobs_queue
        s_queue = [queue[0]]
        for j in sorted(queue[1:], key=lambda j: j.walltime * j.res):
            if self.max_queue_sz and len(s_queue) == max_queue_sz:
                break
            s_queue.append(j)
        return np.asarray(s_queue)

    def on_simulation_begins(self, timestamp, data):
        super().on_simulation_begins(timestamp, data)
        self.nb_nodes = self.platform.nb_nodes

    def set_platform_size(self, sz):
        assert 0 <= self.sz <= self.platform.nb_nodes, 'Cannot expand beyond platform size'
        self.nb_nodes = sz

    def get_available_resources(self, include_omitted=False):
        res = super().get_available_resources()
        res = sorted(res, key=lambda r: r.state.value)
        if include_omitted:
            return res

        nb_res = self.nb_nodes * self.rjms.platform.nodes[0].nb_resources
        return res[:nb_res]

    def get_available_nodes(self, include_omitted=False):
        nodes = super().get_available_nodes()
        nodes = sorted(nodes, key=lambda n: max(
            r.state.value for r in n.resources))
        if include_omitted:
            return nodes

        return nodes[:self.nb_nodes]

    def get_reserved_time(self, include_omitted=False):
        reserved = super().get_reserved_time()
        reserved.sort()
        if include_omitted:
            return reserved

        nb_res = self.nb_nodes * \
            self.rjms.platform.nodes[0].nb_resources  # Temp fix
        return reserved[nb_res:]

    def get_progress(self, include_omitted=False):
        reserved = super().get_progress()
        reserved.sort()
        if include_omitted:
            return reserved

        nb_res = self.nb_nodes * \
            self.rjms.platform.nodes[0].nb_resources  # Temp fix
        return reserved[nb_res:]


class OffReservationEnv(GridEnv):
    def __init__(self,
                 use_batsim=False,
                 simulation_time=1440,
                 files_dir=None,
                 export=False,
                 max_queue_sz=10,
                 act_interval=1,
                 qos_stretch=0.5):

        self.act_interval = act_interval
        self.max_queue_sz = max_queue_sz

        super().__init__(
            use_batsim=use_batsim,
            simulation_time=simulation_time,
            files_dir=files_dir,
            export=export,
            qos_stretch=qos_stretch)

        self.scheduler = EASYBackfilling(self.rjms)

    def _get_rjms(self, use_batsim):
        return RJMSWrapper(self.max_queue_sz, use_batsim)

    def step(self, action):
        assert self.rjms.is_running, "Simulation is not running."

        self._set_off_reservation_size(action)

        jobs_to_start = self.scheduler.schedule()

        self.rjms.start_ready_jobs()

        # This should occur before proceeding time because new jobs can be submitted.
        reward = self._get_reward()

        self.rjms.proceed_time(self.rjms.current_time + self.act_interval)
        if self.rjms.is_running:
            self.rjms.start_ready_jobs()

        obs = self._get_obs(reward)
        done = not self.rjms.is_running
        info = self._get_info()
        return obs, reward, done, info

    def _set_off_reservation_size(self, size):
        if size != 0:
            # delimit the reservation size
            nodes_available = self.rjms.get_available_nodes(
                include_omitted=True)
            size = min(size, len(nodes_available))

            # Just update the scheduler to see if we delay the priority job
            pjob = self.rjms.jobs_queue[0] if self.rjms.queue_lenght > 0 else None
            # free some resources for priority job
            if pjob and pjob.expected_time_to_start == 0:
                nb_nodes_needed = math.ceil(
                    pjob.res / self.rjms.platform.nodes[0].nb_resources)
                nb_nodes_free = self.rjms.platform.nb_nodes - nb_nodes_needed
                size = min(size, nb_nodes_free)

            # find how many nodes must be turned off
            nodes_on = [n.id for n in nodes_available if n.is_on]
            if nodes_on:
                nb_not_on = len(nodes_available) - len(nodes_on)
                nb_to_turn_off = max(0, size - nb_not_on)
                self.rjms.turn_off(*nodes_on[:nb_to_turn_off])

        size = self.rjms.platform.nb_nodes - size
        self.rjms.set_platform_size(size)

    def _get_reward(self):
        energy_waste = 0
        for n in self.rjms.platform.nodes:
            p_max = max(ps.power_min for ps in n.power_states)
            if n.is_switching_off or n.is_switching_on or n.is_idle:
                energy_waste += n.power / p_max
        energy_waste /= self.rjms.platform.nb_nodes

        queue, qos = self.rjms.jobs_queue, 0
        if self.rjms.nb_nodes != self.rjms.platform.nb_nodes and len(queue) > 0:
            agenda = self.rjms.get_reserved_time(include_omitted=True)
            resources = self.rjms.get_available_resources(include_omitted=True)
            scheduler_plan = self.scheduler.plan(queue, resources, agenda)
            for job, alloc in scheduler_plan:
                if (self.rjms.current_time - job.subtime) / job.walltime >= self.qos_stretch:
                    qos += job.res
            qos /= self.rjms.platform.nb_resources

        return -1 * (energy_waste + qos)

    def _get_obs(self, reward=0):
        obs = {}

        # Update scheduler
        agenda = self.rjms.get_reserved_time()
        resources = self.rjms.get_available_resources()
        self.scheduler.plan(self.rjms.jobs_queue, resources, agenda)

        obs['queue'] = np.asarray(
            [
                [j.subtime, j.res, j.walltime, j.expected_time_to_start, j.user, int(j.profile)] for j in self.rjms.jobs_queue
            ],
            dtype=self.observation_space.spaces['queue'].dtype
        )

        obs['jobs_running'] = np.asarray(
            [
                [j.start_time, j.res, j.walltime, j.user, int(j.profile)] for j in self.rjms.jobs_running
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

        for i, p in enumerate(self.rjms.get_progress(include_omitted=True)):
            obs['agenda'][i] = p

        obs['reservation_size'] = self.rjms.platform.nb_nodes - self.rjms.nb_nodes
        obs['time'] = self.rjms.current_time
        obs['reward'] = reward
        return obs

    def _get_space(self):
        self._start_simulation()

        queue = spaces.Box(
            low=-1,
            high=np.iinfo(int).max,
            shape=(self.max_queue_sz, 6),
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
