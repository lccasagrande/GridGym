import os
import math

import gym
from gym import error
from gym.utils import seeding

from gridgym.envs.simulator.manager import ResourceManager
from gridgym.envs.simulator.utils.monitors import *
from gridgym.envs.simulator.utils.submitter import JobSubmitter


class GridEnv(gym.Env):
    SIMULATION_TIME = None  # 7 * 1440
    WORKLOADS = 'GridGym/gridgym/envs/simulator/files/workloads/'
    PLATFORM = 'GridGym/gridgym/envs/simulator/files/platform.xml'
    OUTPUT = 'GridGym/gridgym/envs/simulator/files/output/'

    def __init__(self, use_batsim=False):
        self.rjms = ResourceManager(use_batsim)
        self.job_submitter = JobSubmitter(self.rjms.simulator)

        self.workloads = []
        for f in os.scandir(self.WORKLOADS):
            if f.is_dir():
                group_workloads = [os.path.join(
                    f, w) for w in os.listdir(f) if w.endswith('.json')]
                self.workloads.append(group_workloads)

        if len(self.workloads) == 0:
            self.workloads = [os.path.join(self.WORKLOADS, w) for w in os.listdir(
                self.WORKLOADS) if w.endswith('.json')]
        elif len(self.workloads) == 1:
            self.workloads = self.workloads[0]

        assert len(self.workloads) > 0

        #self.workloads.sort(key=lambda w: int(w[w.rfind("_")+1:w.rfind(".json")]))
        self.observation_space, self.action_space = self._get_space()
        self.seed()

    def close(self):
        self.rjms.close()
        self.job_submitter.close()

    def reset(self):
        self.rjms.close()
        self.rjms.start(platform_fn=self.PLATFORM,
                        output_dir=self.OUTPUT,
                        simulation_time=self.SIMULATION_TIME)

        group = self.np_random.choice(self.workloads)
        # self.np_random.shuffle(self.workloads)
        w = self.np_random.choice(group) if isinstance(group, list) else group
        self.job_submitter.start(w)
        return self._get_obs()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError

    def _get_space(self):
        raise NotImplementedError

    def _get_obs(self):
        raise NotImplementedError
