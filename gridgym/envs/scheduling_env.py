import math
import sys
from collections import defaultdict, deque

import numpy as np
import gym
from gym import spaces
from procset import ProcSet

from .grid_env import GridEnv
from batsim_py.utils.shutdown import Timeout
from batsim_py.rjms import RJMSHandler, InsufficientResources
from batsim_py.utils.schedulers import EASYBackfilling
from batsim_py.utils.monitors import NodeStatsMonitor
from batsim_py.resource import ResourceState, PowerStateType


class RJMSWrapper(RJMSHandler):
    def __init__(self, timeout, use_batsim=False):
        super().__init__(use_batsim=use_batsim)
        self.users = defaultdict(lambda: deque([1], maxlen=20))
        self._timeout = Timeout(timeout, self)

    def get_confidence(self, user):
        return np.mean(self.users[user])

    def on_job_completed(self, timestamp, data):
        super().on_job_completed(timestamp, data)
        job = self._jobs['completed'][-1]
        self.users[job.user].append(job.runtime / job.walltime)
        assert job.id == data.job_id


class SchedulingEnv(GridEnv):
    def __init__(self,
                 use_batsim=False,
                 files_dir=None,
                 export=False,
                 max_queue_sz=20,
                 timeout=15,
                 act_interval=1,
                 max_simulation_time=None):

        self.act_interval = act_interval
        self.max_queue_sz = max_queue_sz
        self.timeout = timeout
        self.max_sim_time = max_simulation_time  # simulation_time
        super().__init__(use_batsim=use_batsim, files_dir=files_dir, export=export)

    def _get_rjms(self, use_batsim):
        return RJMSWrapper(timeout=self.timeout, use_batsim=use_batsim)

    def reset(self):
        super().reset()
        while self.rjms.is_running and self.rjms.queue_lenght == 0:
            self.rjms.proceed_time()
        return self._get_obs()

    def _proceed_time(self):
        self.rjms.proceed_time(self.rjms.current_time + self.act_interval)
        while self.rjms.is_running and self.rjms.queue_lenght == 0 and all(n.is_off for n in self.rjms.platform.nodes):
            self.rjms.proceed_time()

    def step(self, action):
        assert self.rjms.is_running, "Simulation is not running."

        reward = 0
        if 0 < action <= self.rjms.queue_lenght:
            try:
                job = self.rjms.jobs_queue[action-1]
                self.rjms.allocate(job.id)
            except InsufficientResources:
                self.rjms.start_ready_jobs()
                reward = self._get_reward()
                self._proceed_time()
        else:
            self.rjms.start_ready_jobs()
            reward = self._get_reward()
            self._proceed_time()

        if self.max_sim_time and self.rjms.current_time >= self.max_sim_time:
            self.rjms.close()

        obs = self._get_obs()
        done = not self.rjms.is_running
        info = self._get_info()
        return obs, reward, done, info

    def _get_reward(self, job=None):
        # QoS
        wait_t = sum(1./j.walltime for j in self.rjms.jobs_queue[:self.max_queue_sz])

        # Energy waste
        energy_score = sum(1 for n in self.rjms.platform.nodes if n.is_idle)
        energy_score /= self.rjms.platform.nb_nodes

        # Utilization
        u = sum(
            1 for n in self.rjms.platform.nodes for r in n.resources if r.is_computing)
        u /= self.rjms.platform.nb_resources

        u_weight = 1 / self.timeout
        e_weight = -1
        qos_weight = -1
        return (e_weight * energy_score) + (u_weight * u) + (qos_weight * wait_t)

    def _get_obs(self):
        obs = {}

        # Queue State
        queue = self.rjms.jobs_queue
        obs['queue'] = {}
        obs['queue']['lenght'] = len(queue)
        obs['queue']['load'] = sum(j.res * j.walltime for j in queue)
        obs['queue']['jobs'] = np.full(
            fill_value=None,
            shape=self.observation_space.spaces['queue']['jobs'].shape,
            dtype=self.observation_space.spaces['queue']['jobs'].dtype)

        for i, j in enumerate(queue[:self.max_queue_sz]):
            job_state = {
                'subtime': j.subtime,
                'res': j.res,
                'walltime': j.walltime,
                'expected_time_to_start': j.expected_time_to_start,
                'user': j.user,
                'profile': int(j.profile),
                'confidence': self.rjms.get_confidence(j.user)
            }
            obs['queue']['jobs'][i] = job_state

        obs['platform'] = {}
        obs['platform']['jobs'] = np.full(
            fill_value=None,
            shape=self.observation_space.spaces['platform']['jobs'].shape,
            dtype=self.observation_space.spaces['platform']['jobs'].dtype)

        for i, j in enumerate(self.rjms.jobs_running):
            job_state = {
                'start_time': j.start_time,
                'res': j.res,
                'allocation': str(ProcSet(*j.allocation)),
                'walltime': j.walltime,
                'user': j.user,
                'profile': int(j.profile),
                'confidence': self.rjms.get_confidence(j.user)
            }
            obs['platform']['jobs'][i] = job_state

        obs['platform']['nb_reserved'] = self.rjms.platform.nb_resources - \
            len(self.rjms.get_available_resources())
        obs['platform']['status'] = np.zeros(
            shape=self.observation_space.spaces['platform']['status'].shape,
            dtype=self.observation_space.spaces['platform']['status'].dtype)

        for n in self.rjms.platform.nodes:
            obs['platform']['status'][int(n.id)] = [
                r.state.value for r in n.resources]

        obs['agenda'] = np.zeros(
            shape=self.observation_space.spaces['agenda'].shape,
            dtype=self.observation_space.spaces['agenda'].dtype)

        for i, p in enumerate(self.rjms.get_reserved_time()):
            obs['agenda'][i] = p

        obs['time'] = self.rjms.current_time
        return obs

    def _get_space(self):
        self._start_simulation()

        queue = spaces.Dict({
            'lenght': spaces.Discrete(np.iinfo(int).max),
            'load': spaces.Discrete(np.iinfo(int).max),
            'jobs': spaces.Box(
                low=-1,
                high=np.iinfo(int).max,
                shape=(self.max_queue_sz,),
                dtype=np.object)
        })

        platform = spaces.Dict({
            'nb_reserved': spaces.Discrete(self.rjms.platform.nb_resources + 1),
            'status': spaces.Box(
                low=0,
                high=5,
                shape=(self.rjms.platform.nb_nodes,
                       max(n.nb_resources for n in self.rjms.platform.nodes)),
                dtype=np.float),
            'jobs': spaces.Box(
                low=-1,
                high=np.iinfo(int).max,
                shape=(self.rjms.platform.nb_resources, ),
                dtype=np.object)
        })

        agenda = spaces.Box(
            low=0,
            high=1,
            shape=(self.rjms.platform.nb_resources,),
            dtype=np.float)

        space = spaces.Dict({
            'queue': queue,
            'platform': platform,
            'agenda': agenda,
            'time': spaces.Box(low=0, high=np.iinfo(int).max, shape=(), dtype=np.int)
        })
        act_space = spaces.Discrete(self.max_queue_sz + 1)
        self.rjms.close()
        return space, act_space

    def _get_info(self):
        info = {'workload_name': self.workload_name}
        return info
