import math
import sys

import numpy as np
import gym
from gym import spaces

from .grid_env import GridEnv
from batsim_py.utils.schedulers import EASYBackfilling
from batsim_py.resource import ResourceState, PowerStateType


class SchedulingEnv(GridEnv):
    MAX_QUEUE_SZ = 20
    ACT_INTERVAL = 1  # minutes
    SIMULATION_TIME = None  # 60 * 60 * 24  # minutes

    def step(self, action):
        assert self.rjms.is_running, "Simulation is not running."

        if 0 < action <= len(self.rjms.jobs_queue):
            try:
                self.rjms.allocate(self.rjms.jobs_queue[action-1].id)
            except:
                self.rjms.proceed_time(
                    self.rjms.current_time + self.ACT_INTERVAL)
        else:
            self.rjms.proceed_time(self.rjms.current_time + self.ACT_INTERVAL)

        obs = self._get_obs()
        reward = self._get_reward()
        done = not self.rjms.is_running
        #info = {
        #    k: v for k, v in self.scheduler_monitor.info.items()
       # } if done else {}
        return obs, reward, done, {}

    def _get_reward(self):
        penalty = sum(1 / (j.walltime / 2.)
                      for j in self.rjms.jobs_queue[:self.MAX_QUEUE_SZ])
        return -1 * penalty

    def _get_obs(self):
        obs = {}
        obs['queue'] = np.full(
            fill_value=-1,
            shape=self.observation_space.spaces['queue'].shape,
            dtype=self.observation_space.spaces['queue'].dtype)
        for i, j in enumerate(self.rjms.jobs_queue[:self.MAX_QUEUE_SZ]):
            time_to_start = -1 if j.expected_time_to_start == - \
                1 else max(0, j.expected_time_to_start - self.ACT_INTERVAL)
            obs['queue'][i] = np.asarray(
                [j.subtime, j.res, j.walltime, time_to_start])

        obs['platform'] = np.zeros(
            shape=self.observation_space.spaces['platform'].shape,
            dtype=self.observation_space.spaces['platform'].dtype)
        for i, n in enumerate(self.rjms.platform.nodes):
            obs['platform'][i] = np.asarray(
                [r.state.value for r in n.resources])

        obs['agenda'] = np.zeros(
            shape=self.observation_space.spaces['agenda'].shape,
            dtype=self.observation_space.spaces['agenda'].dtype)

        for i, r in enumerate(self.rjms.platform.resources):
            if r.is_computing:
                obs['agenda'][i] = np.asarray(
                    [r.allocated_job.start_time, r.allocated_job.start_time + r.allocated_job.walltime])
            elif r.is_reserved:
                obs['agenda'][i] = np.asarray(
                    [self.rjms.current_time, self.rjms.current_time + r.allocated_job.walltime])

        obs['time'] = self.rjms.current_time
        return obs

    def _get_space(self):
        self.rjms.start(self.PLATFORM, self.workloads[0],
                        self.OUTPUT, self.SIMULATION_TIME)

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
            high=np.iinfo(int).max,
            shape=(self.rjms.platform.nb_resources, 2),
            dtype=np.int)
        obs_space = spaces.Dict({
            'queue': queue,
            'platform': platform,
            'agenda': agenda,
            'time': spaces.Box(low=0, high=np.iinfo(int).max, shape=(), dtype=np.int)
        })
        act_space = spaces.Discrete(self.MAX_QUEUE_SZ + 1)
        self.rjms.close()
        return obs_space, act_space
