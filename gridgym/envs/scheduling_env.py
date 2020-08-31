from collections import defaultdict
from collections import deque
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Dict

from batsim_py import HostEvent
from batsim_py import SimulatorEvent
import batsim_py
from batsim_py.resources import Host
import gym
from gym import spaces
from gym import error
import numpy as np

from .grid_env import GridEnv


class ShutdownPolicy():
    def __init__(self, timeout: int, simulator: batsim_py.SimulatorHandler):
        super().__init__()
        self.timeout = timeout
        self.simulator = simulator
        self.idle_servers: Dict[int, float] = {}

        self.simulator.subscribe(
            HostEvent.STATE_CHANGED, self._on_host_state_changed)
        self.simulator.subscribe(
            SimulatorEvent.SIMULATION_BEGINS, self._on_sim_begins)

    def shutdown_idle_hosts(self, *args, **kwargs):
        hosts_to_turnoff = []
        for h_id, start_t in list(self.idle_servers.items()):
            if self.simulator.current_time - start_t >= self.timeout:
                hosts_to_turnoff.append(h_id)
                del self.idle_servers[h_id]

        if hosts_to_turnoff:
            self.simulator.switch_off(hosts_to_turnoff)

    def _on_host_state_changed(self, host: Host):
        if host.is_idle:
            if host.id not in self.idle_servers:
                self.idle_servers[host.id] = self.simulator.current_time
                t = self.simulator.current_time + self.timeout
                self.simulator.set_callback(t, self.shutdown_idle_hosts)
        else:
            self.idle_servers.pop(host.id, None)

    def _on_sim_begins(self, _):
        self.idle_servers.clear()
        for h in self.simulator.platform.hosts:
            if h.is_idle:
                self.idle_servers[h.id] = self.simulator.current_time
                t = self.simulator.current_time + self.timeout
                self.simulator.set_callback(t, self.shutdown_idle_hosts)


class SchedulingEnv(GridEnv):
    def __init__(self,
                 platform_fn: str,
                 workloads_dir: str,
                 t_action: int = 1,
                 t_shutdown: int = 0,
                 hosts_per_server: int = 1,
                 queue_max_len: int = 20,
                 seed: Optional[int] = None,
                 external_events_fn: Optional[str] = None,
                 simulation_time: Optional[float] = None) -> None:

        if t_action < 0:
            raise error.Error('Expected `t_action` argument to be greater '
                              f'than zero, got {t_action}.')

        self.user_conf: dict = defaultdict(lambda: deque([1], maxlen=20))
        self.queue_max_len = queue_max_len
        self.t_action = t_action

        super().__init__(platform_fn, workloads_dir, seed,
                         external_events_fn, simulation_time, True,
                         hosts_per_server=hosts_per_server)
        self.simulator.subscribe(
            batsim_py.JobEvent.COMPLETED, self._on_job_completed)

        self.shutdown_policy = ShutdownPolicy(t_shutdown, self.simulator)

    def _on_job_completed(self, job: batsim_py.jobs.Job) -> None:
        if job.user_id is not None:
            self.user_conf[job.user_id].append(job.runtime / job.walltime)

    def step(self, action: int) -> Tuple[Any, float, bool, dict]:
        assert self.simulator.is_running and self.simulator.platform

        if self.queue_max_len < action < 0:
            raise error.InvalidAction(f'Invalid action {action}.')

        scheduled, reward = False, 0.
        if action > 0:
            job = self.simulator.queue[action-1]
            available = self.simulator.platform.get_not_allocated_hosts()
            if job.res <= len(available):
                res = [h.id for h in available[:job.res]]
                self.simulator.allocate(job.id, res)
                scheduled = True

        if not scheduled:
            reward = self._get_reward()
            self.simulator.proceed_time(self.t_action)

        obs = self._get_state()
        done = not self.simulator.is_running
        info = {"workload": self.workload}
        return obs, reward, done, info

    def _get_reward(self) -> float:
        nb_hosts = len(list(self.simulator.platform.hosts))
        # QoS
        wait_t = sum(
            1./j.walltime for j in self.simulator.queue[:self.queue_max_len])

        # Energy waste
        energy_score = sum(
            1. for h in self.simulator.platform.hosts if h.is_idle)
        energy_score /= nb_hosts

        # Utilization
        u = sum(1. for h in self.simulator.platform.hosts if h.is_computing)
        u /= nb_hosts

        u_weight = 1 / max(1, self.shutdown_policy.timeout)
        e_weight = -1
        qos_weight = -1
        return (e_weight * energy_score) + (u_weight * u) + (qos_weight * wait_t)

    def _get_state(self) -> Any:
        state: dict = {}

        # Queue
        queue: dict = {}
        queue['size'] = len(self.simulator.queue)
        queue['jobs'] = np.full(fill_value=0, shape=(self.queue_max_len, 5))
        for i, job in enumerate(self.simulator.queue[:self.queue_max_len]):
            conf = - \
                1 if job.user_id is None else np.mean(
                    self.user_conf[job.user_id])
            wall = -1 if job.walltime is None else job.walltime
            user = -1 if job.user_id is None else job.user_id

            queue['jobs'][i] = [
                job.subtime,
                job.res,
                wall,
                user,
                conf
            ]
        # Platform
        nb_hosts = len(list(self.simulator.platform.hosts))
        platform: dict = {}
        platform['status'] = np.array(
            [h.state.value for h in self.simulator.platform.hosts])
        platform['agenda'] = np.full(fill_value=0, shape=(nb_hosts, 3))
        for j in self.simulator.jobs:
            if j.is_running:
                for h_id in j.allocation:
                    platform['agenda'][h_id] = [
                        j.start_time,
                        j.walltime or -1,
                        j.user_id or -1,
                    ]

        state['queue'] = queue
        state['platform'] = platform
        state['current_time'] = self.simulator.current_time
        return state

    def _get_spaces(self) -> Tuple[spaces.Dict, spaces.Discrete]:
        agenda_shape = status_shape = ()
        if self.simulator.is_running:
            nb_hosts = len(self.hosts)
            status_shape = (nb_hosts, )
            agenda_shape = (nb_hosts, 3)

        queue = spaces.Dict({
            'size': spaces.Discrete(float('inf')),
            'jobs': spaces.Box(low=-1,
                               high=float('inf'),
                               shape=(self.queue_max_len, 5))
        })

        platform = spaces.Dict({
            'agenda': spaces.Box(low=-1, high=float('inf'), shape=agenda_shape),
            'status': spaces.Box(low=0, high=7, shape=status_shape)
        })

        obs_space = spaces.Dict({
            'queue': queue,
            'platform': platform,
            'current_time': spaces.Box(low=0, high=float('inf'), shape=())
        })

        act_space = spaces.Discrete(self.queue_max_len + 1)
        return obs_space, act_space
