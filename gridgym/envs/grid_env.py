import os
import math
import shutil
import importlib.resources as impr
import atexit
import signal
import sys

from contextlib import ExitStack

import gym
from gym import error
from gym.utils import seeding

import gridgym.envs.simulator as sim_dir
from gridgym.envs.simulator.manager import ResourceManager
from gridgym.envs.simulator.utils.monitors import *
from gridgym.envs.simulator.utils.commons import *
from gridgym.envs.simulator.utils.submitter import JobSubmitter


class GridEnv(gym.Env):
    OUTPUT = "/tmp/GridGym/"

    def __init__(self, use_batsim=False, simulation_time=None):
        self.simulation_time = simulation_time
        self.workloads = self.platform_fn = self.file_manager = None
        self.rjms = ResourceManager(use_batsim)
        self.job_submitter = JobSubmitter(self.rjms.simulator)
        atexit.register(self.close)
        signal.signal(signal.SIGTERM, signal_wrapper(self.close))
        self._load()
        self.observation_space, self.action_space = self._get_space()
        self.seed()
        self.metadata = {'render.modes': []}


    def _load(self):
        self.file_manager = ExitStack()
        path = self.file_manager.enter_context(impr.path(sim_dir, 'files'))
        platform_fn = "{}/platform.xml".format(path)
        workloads_path = "{}/workloads/".format(path)
        workloads = []

        for f in os.scandir(workloads_path):
            if f.is_dir():
                group_workloads = [
                    os.path.join(f, w) for w in os.listdir(f) if w.endswith('.json')
                ]
                if len(group_workloads) > 0:
                    workloads.append(group_workloads)
        if len(workloads) == 0:
            workloads = [
                os.path.join(workloads_path, w) for w in os.listdir(workloads_path) if w.endswith('.json')
            ]
        elif len(workloads) == 1:
            workloads = workloads[0]

        self.workloads = workloads
        self.platform_fn = platform_fn

    def _start_simulation(self):
        overwrite_dir(self.OUTPUT)
        self.rjms.close()
        self.rjms.start(platform_fn=self.platform_fn,
                        output_dir=self.OUTPUT,
                        simulation_time=self.simulation_time)

    def close(self, s=None, a=None):
        self.rjms.close()
        self.job_submitter.close()
        if self.file_manager is not None:
            self.file_manager.close()
        self.workloads = self.platform_fn = self.file_manager = None

    def reset(self):
        assert len(self.workloads) > 0
        self._start_simulation()
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
