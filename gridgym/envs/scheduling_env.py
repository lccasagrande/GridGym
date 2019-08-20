import math
import sys

import numpy as np
import gym
from gym import spaces

from .grid_env import GridEnv
from .simulator.simulator import SimulationHandler
from .simulator.schedulers import EASYBackfilling
from .simulator.resource import ResourceState, PowerStateType


class SchedulingEnv(GridEnv):
    SIMULATION_TIME = None
    QUEUE_SIZE = 50

    def step(self, action):
        assert self.simulator.is_running, "Simulation is not running."

        if action > 0 and action <= len(self.simulator.jobs_queue):
            try:
                self.simulator.allocate(self.simulator.jobs_queue[action-1].id)
            except:
                self.simulator.proceed_time()
        else:
            self.simulator.proceed_time()

        # This should occur before proceeding time because new jobs can be submitted.
        obs = self._get_obs()
        reward = self._get_reward()
        done = not self.simulator.is_running
        info = {}

        return obs, reward, done, info

    def _get_reward(self):
        return -1 * len(self.simulator.jobs_queue)

    def _get_obs(self):
        obs = {}
        obs['queue'] = np.asarray([(j.res, j.walltime)
                                   for j in self.simulator.jobs_queue])
        obs['platform'] = np.asarray(
            [(r.parent_id, r.state.value) for r in self.simulator.platform.resources])
        obs['agenda'] = np.zeros(
            shape=(self.simulator.platform.nb_resources, ),
            dtype=np.int)

        for j in self.simulator.jobs_running:
            obs['agenda'][j.allocation] = (
                j.walltime + j.start_time) - self.simulator.current_time
        obs['time'] = self.simulator.current_time
        return obs

    def _get_space(self):
        if not self.simulator.platform:
            self.reset()

        queue = spaces.Box(
            low=np.asarray([1, 1]),
            high=np.asarray(
                [self.simulator.platform.nb_resources, sys.maxsize]),
            dtype=np.int)
        platform = spaces.Box(
            low=np.asarray([0, 0]),
            high=np.asarray(
                [self.simulator.platform.nb_nodes, len(ResourceState)]),
            dtype=np.int)
        agenda = spaces.Box(
            low=0,
            high=sys.maxsize,
            shape=(self.simulator.platform.nb_resources,),
            dtype=np.int)
        obs_space = spaces.Dict({
            'queue': queue,
            'platform': platform,
            'agenda': agenda,
            'time': spaces.Discrete(sys.maxsize)
        })
        act_space = spaces.Discrete(self.QUEUE_SIZE + 1)
        return obs_space, act_space

if __name__ == '__main__':
    from shutil import which
    print(which('batsim'))