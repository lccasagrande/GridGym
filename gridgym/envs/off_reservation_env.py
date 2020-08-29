import decimal
import math
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Any
from typing import Optional

from batsim_py.jobs import Job
from batsim_py.monitors import ConsumedEnergyMonitor
from batsim_py.monitors import HostStateSwitchMonitor
from batsim_py.monitors import JobMonitor
from batsim_py.monitors import SimulationMonitor
from batsim_py.resources import Host
from batsim_py.simulator import Reservation
from evalys.visu.legacy import plot_mstates
from gym import error
import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np

from .grid_env import GridEnv


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


class OffReservationEnv(GridEnv):
    def __init__(self,
                 platform_fn: str,
                 workloads_dir: str,
                 t_action: int = 1,
                 queue_max_len: int = 20,
                 hosts_per_server: int = 1,
                 qos_treshold=0.5,
                 seed: Optional[int] = None,
                 external_events_fn: Optional[str] = None,
                 simulation_time: Optional[float] = None) -> None:

        self.queue_max_len = queue_max_len
        self.t_action = t_action
        self.qos_treshold = qos_treshold
        self.hosts_per_server = hosts_per_server
        self.reserved_servers: Dict[int, Server] = {}
        self.servers: List[Server] = []
        self.hosts: List[Host] = []

        super().__init__(platform_fn, workloads_dir, seed,
                         external_events_fn, simulation_time, True)

        self.jobs_mon = JobMonitor(self.simulator)
        self.sim_mon = SimulationMonitor(self.simulator)
        self.host_mon = HostStateSwitchMonitor(self.simulator)
        self.e_mon = ConsumedEnergyMonitor(self.simulator)

        self._start_simulator()
        self._load_servers()
        self._close_simulator()
        self.observation_space, self.action_space = self._get_spaces()

    def reset(self) -> Any:
        self._close_simulator()
        self._start_simulator()
        self._load_servers()
        return self._get_state()

    def step(self, action: int) -> Tuple[Any, float, bool, dict]:
        if not self.simulator.is_running or not self.simulator.platform:
            raise error.ResetNeeded("Simulation not running.")

        if 0 > action > len(self.servers):
            raise error.InvalidAction(f'Invalid Action: {action} .')

        # Reserve servers (nodes)
        self._set_reservation(action)

        # Schedule jobs
        if self.simulator.queue:
            # We cannot delay the priority job
            p_job = self.simulator.queue[0]
            l_agenda = self._get_local_agenda()
            p_start_t, _ = self._get_next_available_hosts(l_agenda, p_job.res)
            nb_req_servers = math.ceil(p_job.res / self.hosts_per_server)
            reserved = [s for s in self.servers if s.is_reserved]
            nb_avail = len(self.servers) - len(reserved)
            if p_start_t == 0 and nb_req_servers > nb_avail:
                # Shrink reservation size
                shrink_sz = nb_req_servers - nb_avail
                while shrink_sz > 0 and reserved:
                    reserved.pop(0).release()
                    shrink_sz -= 1

                # Reschedule
                l_agenda = self._get_local_agenda()

            plan, _ = self._get_schedule_plan(l_agenda)
            for job, alloc in plan:
                self.simulator.allocate(job.id, alloc)

        # Shutdown reserved servers
        self._shutdown_reserved_servers()

        # Proceed time
        reward = self._get_reward()
        self.simulator.proceed_time(self.t_action)

        obs = self._get_state()
        done = not self.simulator.is_running
        info = {"workload": self.workload}

        return obs, reward, done, info

    def _load_servers(self) -> None:
        self.hosts = sorted(self.simulator.platform.hosts,
                            key=lambda h: h.id)
        if len(self.hosts) % self.hosts_per_server != 0:
            raise error.Error('All servers must have the same number of hosts '
                              f'per server ({self.hosts_per_server}), the '
                              f'platform has {len(self.hosts)} hosts.')
        self.reserved_servers.clear()
        self.servers.clear()
        for i in range(0, len(self.hosts), self.hosts_per_server):
            server = Server(self.hosts[i:i+self.hosts_per_server])
            self.servers.append(server)

    def _get_state(self) -> Any:
        # Queue Size
        queue: dict = {}
        queue['size'] = len(self.simulator.queue)

        # Queue Promise
        local_agenda = self._get_local_agenda()
        _, t_pjob_start = self._get_schedule_plan(local_agenda)
        queue['promise'] = -1 if t_pjob_start is None else t_pjob_start

        # Queue Jobs
        jobs_shape = self.observation_space['queue']['jobs'].shape
        queue['jobs'] = np.full(fill_value=0, shape=jobs_shape)
        for i, job in enumerate(self._get_queue()):
            wall = -1 if job.walltime is None else job.walltime
            user = -1 if job.user_id is None else job.user_id
            queue['jobs'][i] = [job.subtime, job.res, wall, user]

        # Platform Status
        platform: dict = {}
        status_shape = self.observation_space['platform']['status'].shape
        platform['status'] = np.full(fill_value=0, shape=status_shape)
        for i, server in enumerate(self.servers):
            platform['status'][i] = [h.state.value for h in server]

        # Platform Agenda
        agenda_shape = self.observation_space['platform']['agenda'].shape
        platform['agenda'] = np.full(fill_value=0, shape=agenda_shape)
        for j in self.simulator.jobs:
            if j.is_running:
                for h_id in j.allocation:
                    platform['agenda'][h_id] = [
                        j.start_time,
                        j.walltime or -1,
                        j.user_id or -1,
                    ]

        state = {
            'queue': queue,
            'platform': platform,
            'current_time': self.simulator.current_time,
        }
        return state

    def _get_spaces(self) -> Tuple[spaces.Dict, spaces.Discrete]:
        queue = spaces.Dict({
            'size':  spaces.Discrete(float('inf')),
            'promise':  spaces.Box(low=-1, high=float('inf'), shape=()),
            'jobs': spaces.Box(low=-1,
                               high=float('inf'),
                               shape=(self.queue_max_len, 4))
        })

        platform = spaces.Dict({
            'agenda': spaces.Box(low=-1, high=float('inf'), shape=(len(self.hosts), 3)),
            'status': spaces.Box(low=0, high=7, shape=(len(self.servers), self.hosts_per_server))
        })

        obs_space = spaces.Dict({
            'queue': queue,
            'platform': platform,
            'reservation_size': spaces.Discrete(len(self.servers) + 1),
            'current_time': spaces.Box(low=0, high=float('inf'), shape=())
        })

        act_space = spaces.Discrete(len(self.servers) + 1)
        return obs_space, act_space

    def _get_local_agenda(self) -> Sequence[Reservation]:
        agenda = []
        jobs = self.simulator.jobs
        for server in self.servers:
            if not server.is_reserved:
                for host in server:
                    release_t = 0.
                    for job_id in host.jobs:
                        job = next(j for j in jobs if j.id == job_id)
                        if job.walltime:
                            runtime = 0
                            if job.is_running:
                                assert job.start_time is not None
                                runtime = self.simulator.current_time - job.start_time
                            job_release_t = job.walltime - runtime
                        else:
                            job_release_t = np.inf

                        release_t = max(release_t, job_release_t)
                    agenda.append(Reservation(host, release_t))
        return agenda

    def _shutdown_reserved_servers(self) -> None:
        for server in self.servers:
            if server.is_reserved:
                for host in server:
                    if not host.is_sleeping and not host.is_switching_off:
                        self.simulator.switch_off([host.id])

    def _set_reservation(self, sz: int) -> None:
        reserved = [s for s in self.servers if s.is_reserved]
        nb_reserved = len(reserved)
        if nb_reserved < sz:
            # Reserve some servers
            available = [
                s for s in self.servers if not s.is_allocated and not s.is_reserved
            ]
            for s in available[:sz - nb_reserved]:
                s.reserve()
        elif nb_reserved > sz:
            # Release some servers
            for s in reserved[:nb_reserved - sz]:
                s.release()

    def _get_queue(self) -> Sequence[Job]:
        queue = []
        if self.simulator.queue:
            p_job = self.simulator.queue[0]
            queue = list(self.simulator.queue[1:self.queue_max_len])
            queue.sort(key=lambda j: j.walltime *
                       j.res if j.walltime else float('inf'))
            queue.insert(0, p_job)
        return queue

    def _get_qos(self, job) -> float:
        return (self.simulator.current_time - job.subtime) / job.walltime

    def _get_reward(self) -> float:
        energy_waste = qos = 0.
        # Energy:
        for n in self.simulator.platform.hosts:
            if n.is_switching_off or n.is_switching_on or n.is_idle:
                p_max = max(ps.watt_idle for ps in n.pstates)
                energy_waste += n.power / p_max

        # QoS:
        if any(s.is_reserved for s in self.servers) and self.simulator.queue:
            jobs, _ = self._get_schedule_plan(list(self.simulator.agenda))
            for job, _ in jobs:
                if job.walltime and self._get_qos(job) >= self.qos_treshold:
                    qos += job.res

        # Normalize
        nb_hosts = len(list(self.simulator.platform.hosts))
        energy_waste /= nb_hosts
        qos /= nb_hosts

        # Calculate reward
        r = -1 * (energy_waste + qos)
        return r

    def _get_next_available_hosts(self,
                                  agenda: Sequence[Reservation],
                                  nb_hosts: int) -> Tuple[float, Sequence[int]]:
        next_releases = sorted(agenda, key=lambda a: a.release_time)
        if next_releases:
            last = min(len(next_releases), nb_hosts) - 1
            p_start_t = next_releases[last].release_time
        else:
            p_start_t = 0

        candidates = [
            r.host.id for r in next_releases if r.release_time <= p_start_t
        ]
        return p_start_t, candidates[-nb_hosts:]

    def _get_schedule_plan(self, agenda: Sequence[Reservation]) -> Tuple[Sequence[Tuple[Job, Sequence[int]]], Optional[float]]:
        # With Backfilling
        agenda_cpy = list(agenda)
        p_start_t: Optional[float] = None
        reserved: Sequence[int] = []
        jobs: List[Tuple[Job, Sequence[int]]] = []

        # Start scheduler
        for p_job in self._get_queue():
            h = [a.host for a in agenda_cpy if not a.host.is_computing]
            h_available = sorted(h, key=lambda h: h.state.value, reverse=True)
            h_not_reserved = [h for h in h_available if h.id not in reserved]

            if p_job.res <= len(h_not_reserved):
                # Schedule job on not reserved hosts.
                alloc = h_not_reserved[:p_job.res]
                for host in alloc:
                    reservation = Reservation(host, p_job.walltime or np.inf)
                    i = next(i for i, r in enumerate(
                        agenda_cpy) if r.host.id == host.id)
                    agenda_cpy[i] = reservation

                jobs.append((p_job, [h.id for h in alloc]))
            elif p_start_t is None or not reserved:
                # Reserve hosts for priority job.
                p_start_t, reserved = self._get_next_available_hosts(
                    agenda_cpy, p_job.res)
                if not h_available:
                    break
            elif p_job.walltime and p_job.walltime <= p_start_t and p_job.res <= len(h_available):
                # Schedule job on reserved hosts without delaying p_job.

                alloc = h_available[:p_job.res]
                for host in alloc:
                    reservation = Reservation(host, p_job.walltime or np.inf)
                    i = next(i for i, r in enumerate(
                        agenda_cpy) if r.host.id == host.id)
                    agenda_cpy[i] = reservation

                jobs.append((p_job, [h.id for h in alloc]))

        return jobs, p_start_t
