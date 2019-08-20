import os

import numpy as np
import gym
from gym import error
from gym.utils import seeding

from .simulator.simulator import SimulationHandler


class GridEnv(gym.Env):
    SIMULATION_TIME = None  # 7 * 1440
    WORKLOADS = 'GridGym/gridgym/envs/simulator/files/workloads'
    PLATFORM = 'GridGym/gridgym/envs/simulator/files/platform.xml'
    OUTPUT = 'GridGym/gridgym/envs/simulator/files/output/'

    def __init__(self):
        self.simulator = SimulationHandler()
        self.workloads = [os.path.join(self.WORKLOADS, w) for w in os.listdir(self.WORKLOADS) if w.endswith('.json')]
        self.observation_space, self.action_space = self._get_space()
        self.seed()

    def close(self):
        self.simulator.close()

    def reset(self):
        self.simulator.close()
        self.simulator.start(self._get_workload(), self.PLATFORM, self.OUTPUT, self.SIMULATION_TIME)
        return self._get_obs()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError

    def _get_workload(self):
        return np.random.choice(self.workloads)

    def _get_space(self):
        raise NotImplementedError

    def _get_obs(self):
        raise NotImplementedError
