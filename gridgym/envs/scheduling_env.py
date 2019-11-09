import math
import sys
from collections import defaultdict

import numpy as np
import gym
from gym import spaces
from procset import ProcSet

from .grid_env import GridEnv
from batsim_py.utils.shutdown import Timeout
from batsim_py.rjms import RJMSHandler, InsufficientResources
from batsim_py.utils.schedulers import EASYBackfilling
from batsim_py.resource import ResourceState, PowerStateType


class RJMSWrapper(RJMSHandler):
    def __init__(self, tax, timeout, max_waiting_time=60, use_batsim=False):
        super().__init__(use_batsim=use_batsim)
        self.tax = tax
        self.max_debt = max_waiting_time * tax
        self.account = defaultdict(float)
        self._timeout = Timeout(timeout, self)

    def start(self, platform_fn, workload_fn=None, output_fn=None, simulation_time=None, qos_stretch=None):
        self.account.clear()
        return super().start(platform_fn, workload_fn=workload_fn, output_fn=output_fn, simulation_time=simulation_time, qos_stretch=qos_stretch)

    def proceed_time(self, until=0):
        curr_time = self.current_time

        super().proceed_time(until=until)

        payment = self.tax * (self.current_time - curr_time)
        for user in self.account.keys():
            self.pay_off(user, payment)

    def get_score(self, user):
        return self.account.get(user, 0)

    def loan(self, user, value):
        assert value >= 0
        self.account[user] = min(self.max_debt, self.account[user] + value)

    def pay_off(self, user, value):
        assert value >= 0
        self.account[user] = max(0, self.account[user] - value)

    def on_job_completed(self, timestamp, data):
        super().on_job_completed(timestamp, data)
        job = self._jobs['completed'][-1]
        assert job.id == data.job_id

        if job.runtime <= self._timeout.idling_time:
            t = self._timeout.idling_time - job.runtime
            nodes_id = set(r.parent_id for r in self.platform.get_resources(job.allocation))
            power = sum(n.power for n in self.platform.get_nodes(nodes_id))
            self.loan(job.user, t*power)

    def allocate(self, job_id, resource_ids=None):
        super().allocate(job_id, resource_ids=resource_ids)
        job = next(job for job in self._jobs['ready'] if job.id == job_id)
        resources = self.platform.get_resources(job.allocation)
        for node in self.platform.get_nodes(set(r.parent_id for r in resources)):
            if not node.is_on:
                switch_on_e = node.estimate_energy_to_wakeup()
                switch_off_e = node.estimate_energy_to_shutdown()
                self.loan(job.user, switch_off_e + switch_on_e)


class SchedulingEnv(GridEnv):
    def __init__(self,
                 use_batsim=False,
                 files_dir=None,
                 export=False,
                 max_queue_sz=20,
                 tax=86,
                 timeout=15,
                 act_interval=1):

        self.tax = tax
        self.act_interval = act_interval
        self.max_queue_sz = max_queue_sz
        self.timeout = timeout
        super().__init__(
            use_batsim=use_batsim,
            files_dir=files_dir,
            export=export)

    def _get_rjms(self, use_batsim):
        return RJMSWrapper(tax=self.tax, timeout=self.timeout, use_batsim=use_batsim)

    def reset(self):
        obs = super().reset()
        return obs

    def _proceed_time(self):
        self.rjms.proceed_time(self.rjms.current_time + self.act_interval)
        while self.rjms.is_running and self.rjms.queue_lenght == 0:
            self.rjms.proceed_time()

    def step(self, action):
        assert self.rjms.is_running, "Simulation is not running."

        if 0 < action <= self.rjms.queue_lenght:
            try:
                self.rjms.allocate(self.rjms.jobs_queue[action-1].id)
            except InsufficientResources:
                self._proceed_time()
        else:
            self._proceed_time()

        obs = self._get_obs()
        reward = self._get_reward()
        done = not self.rjms.is_running
        info = self._get_info()
        return obs, reward, done, info

    def _get_reward(self):
        # Waiting time
        wait_score = sum(
            j.res for j in self.rjms.jobs_queue[:self.max_queue_sz] if self.rjms.get_score(j.user) == 0)
        wait_score = min(1., wait_score / self.rjms.platform.nb_resources)

        # Energy waste
        energy_score = sum(
            n.nb_resources for n in self.rjms.platform.nodes if n.is_switching_on)
        energy_score = min(1., energy_score / self.rjms.platform.nb_resources)

        # Reward
        reward = -1 * (energy_score + wait_score)
        return reward

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
                'score': self.rjms.get_score(j.user)
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
                'allocation': str(ProcSet(*j.allocation)),
                'walltime': j.walltime,
                'user': j.user,
                'profile': int(j.profile)
            }
            obs['platform']['jobs'][i] = job_state

        obs['platform']['status'] = np.zeros(
            shape=self.observation_space.spaces['platform']['status'].shape,
            dtype=self.observation_space.spaces['platform']['status'].dtype)

        for n in self.rjms.platform.nodes:
            obs['platform']['status'][int(n.id)] = [
                r.state.value for r in n.resources]

        obs['agenda'] = np.zeros(
            shape=self.observation_space.spaces['agenda'].shape,
            dtype=self.observation_space.spaces['agenda'].dtype)

        for i, p in enumerate(self.rjms.get_progress()):
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
