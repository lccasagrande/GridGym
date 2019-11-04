import math
import sys

import numpy as np
import gym
from gym import spaces

from .grid_env import GridEnv
from batsim_py.utils.schedulers import EASYBackfilling
from batsim_py.resource import ResourceState, PowerStateType


class SchedulingEnv(GridEnv):
    SIMULATION_TIME = None  # 60 * 60 * 24  # minutes

    def __init__(self,
                 use_batsim=False,
                 simulation_time=1440,
                 files_dir=None,
                 export=False,
                 max_queue_sz=20,
                 act_interval=1,
                 qos_stretch=None):

        self.act_interval = act_interval
        self.max_queue_sz = max_queue_sz
        super().__init__(
            use_batsim=use_batsim,
            simulation_time=simulation_time,
            files_dir=files_dir,
            export=export,
            qos_stretch=qos_stretch)

    def step(self, action):
        assert self.rjms.is_running, "Simulation is not running."

        if 0 < action <= len(self.rjms.jobs_queue):
            try:
                self.rjms.allocate(self.rjms.jobs_queue[action-1].id)
            except:
                self.rjms.proceed_time(
                    self.rjms.current_time + self.act_interval)
        else:
            self.rjms.proceed_time(self.rjms.current_time + self.act_interval)

        obs = self._get_obs()
        reward = self._get_reward()
        done = not self.rjms.is_running
        info = self._get_info()
        return obs, reward, done, info

    def _get_reward(self):
        penalty = sum(1 / (j.walltime / 2.)
                      for j in self.rjms.jobs_queue[:self.max_queue_sz])
        return -1 * penalty

    def _get_obs(self):
        obs = {}

        # Update scheduler
        obs['queue'] = np.asarray(
            [
                [j.subtime, j.res, j.walltime, j.expected_time_to_start, j.user, int(j.profile)] for j in self.rjms.jobs_queue[:self.max_queue_sz]
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

        obs['time'] = self.rjms.current_time
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
            'time': spaces.Box(low=0, high=np.iinfo(int).max, shape=(), dtype=np.int)
        })
        act_space = spaces.Discrete(self.rjms.platform.nb_nodes + 1)
        self.rjms.close()
        return obs_space, act_space

    def _get_info(self):
        info = {'workload_name': self.workload_name}
        return info