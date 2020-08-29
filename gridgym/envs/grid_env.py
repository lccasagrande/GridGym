from abc import abstractmethod
import os
from typing import Any
from typing import Tuple
from typing import Optional
from typing import Sequence

import batsim_py
import gym
from gym import error
from gym.utils import seeding


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
        self.allow_compute_sharing = allow_compute_sharing
        self.allow_storage_sharing = allow_storage_sharing
        self.workload: Optional[str] = None
        self.verbosity = verbosity

    def reset(self) -> Any:
        self._close_simulator()
        self._start_simulator()
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
