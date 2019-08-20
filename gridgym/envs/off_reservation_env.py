import math

import numpy as np
import gym
from gym import spaces

from .grid_env import GridEnv
from .simulator.simulator import SimulationHandler
from .simulator.schedulers import EASYBackfilling
from .simulator.resource import ResourceState, PowerStateType


class OffReservationEnv(GridEnv):
    ACT_INTERVAL = 1  # seconds
    SIMULATION_TIME = 1440
    MAX_QUEUE_SZ = 100

    # Metrics
    ENERGY_WASTE_WEIGHT = .4
    HW_DEGRADATION_WEIGHT = 1
    QOS_WEIGHT = .3
    MAX_WAITING_TIME = 30  # seconds

    def __init__(self):
        self.reservation_size = self.nb_switches = 0
        self.scheduler = EASYBackfilling()
        super().__init__()

    def reset(self):
        self.reservation_size = self.nb_switches = 0
        return super().reset()

    def step(self, action):
        assert self.simulator.is_running, "Simulation is not running."

        self._set_off_reservation_size(action)

        jobs_to_start = self.scheduler.schedule(
            self.simulator.jobs_queue, np.sort(self.simulator.agenda)[self.reservation_size * self.simulator.platform.nodes[0].nb_resources:])

        for job_id in jobs_to_start:
            self.simulator.allocate(job_id)

        # This should occur before proceeding time because new jobs can be submitted.
        reward = self._get_reward()

        self.simulator.proceed_time(
            self.simulator.current_time + self.ACT_INTERVAL)

        done = not self.simulator.is_running
        obs = self._get_obs()
        info = {}

        return obs, reward, done, info

    def _get_reward(self):
        energy_waste = nb_switches = qos = 0

        for n in self.simulator.platform.nodes:
            #if n.is_switching_on or n.is_switching_off or n.is_idle:
            #    max_power = max(
            #        ps.idle_power for ps in n.power_states) * n.nb_resources
            #    energy_waste += n.power / max_power
            energy_waste += int(n.is_idle)
            nb_switches += n.nb_state_switches

        energy_waste /= self.simulator.platform.nb_nodes
        nb_switches /= self.simulator.platform.nb_nodes
        hw_degradation = (nb_switches - self.nb_switches)
        self.nb_switches = nb_switches

        if self.reservation_size != 0 and len(self.simulator.jobs_queue) > 0:
            jobs = self.scheduler.schedule(
                self.simulator.jobs_queue,
                self.simulator.agenda
            )
            qos = 0
            for job_id in jobs:
                job = next(
                    j for j in self.simulator.jobs_queue if j.id == job_id)
                if self.simulator.current_time - job.subtime >= self.MAX_WAITING_TIME:
                    qos += job.res
            qos /= self.simulator.platform.nb_resources

        penalty = (energy_waste * self.ENERGY_WASTE_WEIGHT) + \
            (hw_degradation * self.HW_DEGRADATION_WEIGHT) + \
            (qos * self.QOS_WEIGHT)
        return -1 * penalty

    def _set_off_reservation_size(self, size):
        if size == 0:
            self.reservation_size = 0
            return

        # delimit the reservation size
        nodes_available = [n for n in self.simulator.platform.nodes if not any(
            r.is_reserved for r in n.resources)]
        self.reservation_size = min(size, len(nodes_available))

        # Just update the scheduler to see if we delay the priority job
        self.scheduler.schedule(self.simulator.jobs_queue, np.sort(self.simulator.agenda)[
                                self.reservation_size * self.simulator.platform.nodes[0].nb_resources:])
        pjob = self.scheduler.priority_job
        # free some resources for priority job
        if pjob and pjob.expected_time_to_start <= self.ACT_INTERVAL:
            nb_nodes_needed = math.ceil(
                pjob.res / self.simulator.platform.nodes[0].nb_resources)
            nb_nodes_free = self.simulator.platform.nb_nodes - nb_nodes_needed
            self.reservation_size = min(self.reservation_size, nb_nodes_free)

        # find how many nodes must be turned off
        nodes_on = [n.id for n in nodes_available if n.is_on]
        if nodes_on:
            nb_not_on = len(nodes_available) - len(nodes_on)
            nb_to_turn_off = max(0, self.reservation_size - nb_not_on)
            self.simulator.turn_off(*nodes_on[:nb_to_turn_off])

    def _get_obs(self):
        obs = {}
        obs['queue'] = np.full(
            fill_value=-1,
            shape=self.observation_space.spaces['queue'].shape,
            dtype=self.observation_space.spaces['queue'].dtype)
        for i, j in enumerate(self.simulator.jobs_queue):
            time_to_start = -1 if j.expected_time_to_start == - 1 else max(0, j.expected_time_to_start - self.ACT_INTERVAL)
            obs['queue'][i] = np.asarray([j.subtime, j.res, j.walltime, time_to_start])

        obs['platform'] = np.zeros(
            shape=self.observation_space.spaces['platform'].shape,
            dtype=self.observation_space.spaces['platform'].dtype)
        for i, n in enumerate(self.simulator.platform.nodes):
            obs['platform'][i] = np.asarray(
                [r.state.value for r in n.resources])

        obs['agenda'] = np.zeros(
            shape=self.observation_space.spaces['agenda'].shape,
            dtype=self.observation_space.spaces['agenda'].dtype)

        for i, r in enumerate(self.simulator.platform.resources):
            if r.is_computing:
                obs['agenda'][i] = np.asarray([r.allocated_job.start_time, r.allocated_job.start_time + r.allocated_job.walltime])
            elif r.is_reserved:
                obs['agenda'][i] = np.asarray([self.simulator.current_time, self.simulator.current_time + r.allocated_job.walltime])

        obs['reservation_size'] = self.reservation_size
        obs['time'] = self.simulator.current_time
        return obs

    def _get_space(self):
        self.simulator.start(self._get_workload(),
                             self.PLATFORM, self.OUTPUT, self.SIMULATION_TIME)
                             
        queue = spaces.Box(
            low=-1,
            high=np.iinfo(int).max,
            shape=(self.MAX_QUEUE_SZ, 4),
            dtype=np.int)
        platform = spaces.Box(
            low=0,
            high=np.iinfo(int).max,
            shape=(self.simulator.platform.nb_nodes, max(
                n.nb_resources for n in self.simulator.platform.nodes)),
            dtype=np.int)
        agenda = spaces.Box(
            low=0,
            high=np.iinfo(int).max,
            shape=(self.simulator.platform.nb_resources, 2),
            dtype=np.int)
        obs_space = spaces.Dict({
            'queue': queue,
            'platform': platform,
            'agenda': agenda,
            'reservation_size': spaces.Discrete(self.simulator.platform.nb_nodes + 1),
            'time': spaces.Discrete(np.iinfo(int).max)
        })
        act_space = spaces.Discrete(self.simulator.platform.nb_nodes + 1)
        self.simulator.close()
        return obs_space, act_space
