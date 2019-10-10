import math
import itertools
import os

import pandas as pd
import numpy as np
import gym
from gym import spaces


from gridgym.envs.grid_env import GridEnv
from batsim_py.utils.schedulers import EASYBackfilling, SAFBackfilling
from batsim_py.resource import ResourceState, PowerStateType
from batsim_py.utils.monitors import *


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
        self.qos_stretch = qos_stretch
        self.reservation_size = 0

        super().__init__(
            use_batsim=use_batsim,
            simulation_time=simulation_time,
            files_dir=files_dir,
            export=export)

        self.scheduler = EASYBackfilling()
        self.metadata['render.modes'] = []

    def reset(self):
        self.reservation_size = 0
        obs = super().reset()
        return obs

    def step(self, action):
        assert self.rjms.is_running, "Simulation is not running."

        self._set_off_reservation_size(action)

        reserved = self.rjms.get_reserved_time()
        reserved.sort()
        jobs_to_start = self.scheduler.schedule(
            self._get_queue(self.max_queue_sz),
            reserved[self.reservation_size * self.rjms.platform.nodes[0].nb_resources:])

        for job_id in jobs_to_start:
            self.rjms.allocate(job_id)

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

    def _set_off_reservation_size(self, size):
        if size == 0:
            self.reservation_size = 0
            return

        # delimit the reservation size
        nodes_available = self.rjms.get_available_nodes()
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

        queue = self._get_queue(self.max_queue_sz)
        qos = 0
        if self.reservation_size != 0 and len(queue) > 0:
            r = self.rjms.get_reserved_time()
            jobs_ready = self.scheduler.schedule(queue, r)
            for job_id in jobs_ready:
                job = next(j for j in queue if j.id == job_id)
                if (self.rjms.current_time - job.subtime) / job.walltime >= self.qos_stretch:
                    qos += job.res
            qos /= self.rjms.platform.nb_resources
        return -1 * (energy_waste + qos)

    def _get_obs(self, reward=0):
        obs = {}

        # Update scheduler
        reserved = self.rjms.get_reserved_time()
        reserved.sort()
        nb_reserved = self.reservation_size * \
            self.rjms.platform.nodes[0].nb_resources
        self.scheduler.schedule(self._get_queue(
            self.max_queue_sz), reserved[nb_reserved:])

        obs['queue'] = np.asarray(
            [
                [j.subtime, j.res, j.walltime, j.expected_time_to_start, j.user, int(j.profile)] for j in self._get_queue()
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

        for i, p in enumerate(self.rjms.get_progress()):
            obs['agenda'][i] = p

        obs['reservation_size'] = self.reservation_size
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
