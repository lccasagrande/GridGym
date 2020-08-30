from abc import abstractmethod
import os
from typing import Any
from typing import Tuple
from typing import List
from typing import Optional
from typing import Sequence

import batsim_py
from batsim_py.resources import Host
import gym
from gym import error
from gym.utils import seeding


class Server:
    def __init__(self, hosts: List[Host]):
        if not hosts:
            raise ValueError("Expected `hosts` to be a non empty sequence.")
        self.hosts = hosts
        self.size = len(hosts)
        self.__is_reserved = False

    def __iter__(self):
        return iter(self.hosts)

    @property
    def id(self) -> int:
        return self.hosts[0].id

    @property
    def is_off(self) -> bool:
        return all(h.is_sleeping for h in self.hosts)

    @property
    def is_computing(self) -> bool:
        return any(h.is_computing for h in self.hosts)

    @property
    def is_idle(self) -> bool:
        return all(h.is_idle for h in self.hosts)

    @property
    def is_reserved(self) -> bool:
        return self.__is_reserved

    @property
    def is_allocated(self) -> bool:
        return any(h.is_allocated for h in self.hosts)

    def reserve(self) -> None:
        if self.is_allocated or self.is_reserved:
            raise RuntimeError("Cannot reserve a not available server.")

        self.__is_reserved = True

    def release(self) -> None:
        self.__is_reserved = False


class GridEnv(gym.Env):

    metadata: dict = {'render.modes': []}

    def __init__(self,
                 platform_fn: str,
                 workloads_dir: str,
                 seed: Optional[int] = None,
                 external_events_fn: Optional[str] = None,
                 simulation_time: Optional[float] = None,
                 allow_compute_sharing: bool = False,
                 allow_storage_sharing: bool = True,
                 hosts_per_server: int = 1,
                 verbosity: batsim_py.simulator.BatsimVerbosity = 'quiet') -> None:

        if not platform_fn:
            raise error.Error('Expected `platform_fn` argument to be a non '
                              f'empty string, got {platform_fn}.')
        elif not os.path.exists(platform_fn):
            raise error.Error(f"File {platform_fn} does not exist.")
        else:
            self.platform_fn = platform_fn

        if not workloads_dir:
            raise error.Error('Expected `workloads_dir` argument to be a non '
                              f'empty string, got {workloads_dir}.')
        elif not os.path.exists(workloads_dir):
            raise error.Error(f"Directory {workloads_dir} does not exist.")
        else:
            workloads = os.listdir(workloads_dir)
            self.workloads = [
                os.path.join(workloads_dir, w) for w in workloads if w.endswith('.json')
            ]
            if not self.workloads:
                raise error.Error(f"Workloads not found.")

        if external_events_fn and not os.path.exists(external_events_fn):
            raise error.Error(f"File {external_events_fn} does not exist.")

        self.seed(seed)
        self.simulator = batsim_py.SimulatorHandler()
        self.simulation_time = simulation_time
        self.external_events_fn = external_events_fn
        self.hosts_per_server = hosts_per_server
        self.allow_compute_sharing = allow_compute_sharing
        self.allow_storage_sharing = allow_storage_sharing
        self.workload: Optional[str] = None
        self.verbosity: batsim_py.simulator.BatsimVerbosity = verbosity
        self.servers: List[Server] = []
        self.hosts: List[Host] = []
        self.observation_space, self.action_space = self._get_spaces()

    def reset(self) -> Any:
        self._close_simulator()
        self._start_simulator()
        self._load_servers()
        self.observation_space, self.action_space = self._get_spaces()
        return self._get_state()

    def render(self, mode: str = 'human') -> None:
        raise error.Error(f"Not supported.")

    def close(self) -> None:
        self._close_simulator()

    def seed(self, seed: Optional[int] = None) -> Sequence[int]:
        self.np_random, s = seeding.np_random(seed)
        return [s]

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        raise NotImplementedError

    @abstractmethod
    def _get_state(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _get_spaces(self) -> Tuple[Any, Any]:
        raise NotImplementedError

    def _close_simulator(self) -> None:
        self.simulator.close()

    def _start_simulator(self) -> None:
        self.workload = self.np_random.choice(self.workloads)
        self.simulator.start(platform=self.platform_fn,
                             workload=self.workload,  # type:ignore
                             verbosity=self.verbosity,
                             simulation_time=self.simulation_time,
                             allow_compute_sharing=self.allow_compute_sharing,
                             allow_storage_sharing=self.allow_storage_sharing,
                             external_events=self.external_events_fn)

    def _load_servers(self) -> None:
        self.hosts = sorted(self.simulator.platform.hosts,
                            key=lambda h: h.id)
        if len(self.hosts) % self.hosts_per_server != 0:
            raise error.Error('All servers must have the same number of hosts '
                              f'per server ({self.hosts_per_server}), the '
                              f'platform has {len(self.hosts)} hosts.')
        self.servers.clear()
        for i in range(0, len(self.hosts), self.hosts_per_server):
            server = Server(self.hosts[i:i+self.hosts_per_server])
            self.servers.append(server)
