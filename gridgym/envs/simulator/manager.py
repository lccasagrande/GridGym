import subprocess
import time as tm
from itertools import groupby
from copy import copy
from collections import defaultdict
from abc import ABC, abstractmethod

import numpy as np
from sortedcontainers import SortedDict

from gridgym.envs.simulator.resource import Platform, PowerStateType, ResourceState
from gridgym.envs.simulator.job import Job, JobState
from gridgym.envs.simulator.network import EventType, NotifyType, SimulationEventHandler
from gridgym.envs.simulator.utils.submitter import JobSubmitter
from gridgym.envs.simulator.simulator import GridSimulationHandler, BatsimSimulationHandler


class ResourceManager(SimulationEventHandler):
    def __init__(self, use_batsim=False):
        if use_batsim:
            super().__init__(BatsimSimulationHandler())
        else:
            super().__init__(GridSimulationHandler())
        self.simulator.set_callback(
            EventType.JOB_COMPLETED, self.start_ready_jobs)
        self.simulator.set_callback(
            EventType.RESOURCE_STATE_CHANGED, self.start_ready_jobs)
        self.simulator.set_callback(
            EventType.JOB_KILLED, self.start_ready_jobs)

        self._job_submitter = JobSubmitter(self.simulator)
        self.__jobs = {"queue": [], "running": {}, "ready": []}
        self.agenda = None
        self.platform, self.simulation_time, self.submitter_ended = None, None, False

    @property
    def current_time(self):
        return self.simulator.current_time

    @property
    def is_running(self):
        return self.simulator.is_running and (not self.submitter_ended or any(len(v) > 0 for k, v in self.__jobs.items()))

    @property
    def jobs_queue(self):
        return np.array(self.__jobs["queue"])

    @property
    def jobs_running(self):
        return np.asarray(list(self.__jobs["running"].values()))

    @property
    def queue_lenght(self):
        return len(self.__jobs['queue'])

    def start(self, platform_fn, workload_fn=None, output_dir=None, simulation_time=None):
        assert not self.is_running, "A simulation is already running."
        assert not simulation_time or simulation_time > 0
        self.simulation_time = simulation_time
        self.simulator.start(platform_fn, output_dir=output_dir)
        self.__jobs = {"queue": [], "running": {}, "ready": []}
        self.submitter_ended = False
        if workload_fn:
            self._job_submitter.start(workload_fn)

    def close(self):
        if self.simulator.is_running:
            self.simulator.finish()
        self.__jobs = {"queue": [], "running": {}, "ready": []}
        self._job_submitter.close()

    def proceed_time(self, until=0):
        assert self.simulator.is_running, "Cannot proceed if there is no simulator running."
        self.start_ready_jobs()
        if until > 0:
            assert until > self.current_time
            self.simulator.call_me_later(until + 0.001)
            while self.is_running and self.current_time < until:
                self.simulator.proceed_simulation()
        else:
            self.simulator.proceed_simulation()

        self.check_end_of_simulation()

    def check_end_of_simulation(self):
        if self.submitter_ended and all(len(v) == 0 for k, v in self.__jobs.items()) and self.simulator.is_running:
            while self.simulator.is_running:
                self.simulator.proceed_simulation()

    def on_simulation_begins(self, timestamp, data):
        self.platform = data.platform
        self.agenda = Agenda(self.platform)
        if self.simulation_time:
            self.simulator.call_me_later(self.simulation_time)

    def on_simulation_ends(self, timestamp, data):
        self.__jobs = {"queue": [], "running": {}, "ready": []}
        self._job_submitter.close()

    def on_job_submitted(self, timestamp, data):
        if data.job.res > self.platform.nb_resources:
            self.simulator.reject_job(data.job.id)
        else:
            self.__jobs["queue"].append(data.job)

    def on_job_completed(self, timestamp, data):
        job = self.__jobs["running"].pop(data.job_id)
        job.terminate(self.current_time, data.job_state)
        resources = self.platform.get_resources(job.allocation)
        for r in resources:
            r.release()
            self.agenda.release(r.id)

    def on_job_killed(self, timestamp, data):
        for id in data.job_ids:
            job = self.__jobs["running"].pop(data.job_id)
            job.terminate(self.current_time, JobState.COMPLETED_KILLED)
            resources = self.platform.get_resources(job.allocation)
            for r in resources:
                r.release()
                self.agenda.release(r.id)

    def on_requested_call(self, timestamp, data):
        if self.simulation_time and self.current_time >= self.simulation_time:
            if self.is_running:
                self.simulator.finish()
            else:
                self.check_end_of_simulation()

    def on_notify(self, timestamp, data):
        if data.type == NotifyType.no_more_static_job_to_submit:
            self.submitter_ended = True

    def on_resource_power_state_changed(self, timestamp, data):
        resources = self.platform.get_resources(data.resources)
        for r in resources:
            r.set_pstate(data.pstate)

    def set_pstate(self, *node_ids, pstate_id):
        resources_ids = []
        for i in node_ids:
            node = self.platform.get_node(i)
            node.set_pstate(pstate_id)
            resources_ids.extend([r.id for r in node.resources])

        self.simulator.set_resources_pstate(resources_ids, pstate_id)

    def turn_on(self, *node_ids):
        resources_new_state = defaultdict(list)
        for i in node_ids:
            node = self.platform.get_node(i)
            node.wakeup()
            ps_id = next(
                ps.id for ps in node.power_states if ps.type == PowerStateType.computation)
            resources_new_state[ps_id].extend([r.id for r in node.resources])

        for ps_id, resources_ids in resources_new_state.items():
            self.simulator.set_resources_pstate(resources_ids, ps_id)

    def turn_off(self, *node_ids):
        resources_new_state = defaultdict(list)
        for i in node_ids:
            node = self.platform.get_node(i)
            node.sleep()
            ps_id = next(
                ps.id for ps in node.power_states if ps.type == PowerStateType.sleep)
            resources_new_state[ps_id].extend([r.id for r in node.resources])

        for ps_id, resources_ids in resources_new_state.items():
            self.simulator.set_resources_pstate(resources_ids, ps_id)

    def initiate_resources(self, resources):
        ready, nodes_visited = True, {}
        for r in resources:
            if not r.is_idle and r.parent_id not in nodes_visited:
                ready = False
                nodes_visited[r.parent_id] = True
                if r.is_sleeping:
                    self.turn_on(r.parent_id)
        return ready

    def start_ready_jobs(self, timestamp=None, data=None):
        for job in list(self.__jobs['ready']):
            allocated_resources = self.platform.get_resources(job.allocation)
            ready = self.initiate_resources(allocated_resources)
            if ready:
                for r in allocated_resources:
                    r.start_computing()
                job.start(self.current_time)
                self.simulator.execute_job(job.id, job.allocation)
                self.__jobs['ready'].remove(job)
                self.__jobs['running'][job.id] = job

    def allocate(self, job_id, resource_ids=None):
        job = next(j for j in self.__jobs['queue'] if j.id == job_id)
        if resource_ids is None:
            resources = sorted(
                self.agenda.get_available_resources(), key=lambda r: r.state.value)
            resource_ids = [r.id for r in resources[:job.res]]

        if len(resource_ids) != job.res:
            raise Exception("Insufficient resources for job {}".format(job.id))

        job.set_allocation(resource_ids)
        self.agenda.reserve(job)
        self.__jobs['queue'].remove(job)
        self.__jobs['ready'].append(job)


class Agenda():
    def __init__(self, platform):
        self.platform = platform
        self.reservations = SortedDict(lambda k: int(
            k.id), {r: None for r in platform.resources})

    def get_available_nodes(self):
        nodes = []
        for node_id, reservations in groupby(self.reservations.items(), key=lambda it: it[0].parent_id):
            if all(j is None for (r, j) in reservations):
                nodes.append(self.platform.get_node(node_id))
        return np.asarray(nodes)

    def get_available_resources(self):
        return np.asarray([r for r, j in self.reservations.items() if j is None])

    def reserve(self, job):
        assert all(self.reservations[r] is None for r in job.allocation)
        for r in job.allocation:
            self.reservations[r] = job

    def release(self, *resource_ids):
        for r in resource_ids:
            self.reservations[r] = None

    def get_progress(self, current_time):
        remaining = np.zeros(shape=len(self.reservations))
        for i, (r, j) in enumerate(self.reservations.items()):
            if j is None:
                remaining[i] = 0
            elif j.state == JobState.RUNNING:
                runtime = current_time - j.start_time
                remaining[i] = 1 - (runtime / j.walltime)
            else:
                remaining[i] = 1
        return remaining

    def get_reserved_time(self, current_time):
        reserved = np.zeros(shape=len(self.reservations))
        for i, (r, j) in enumerate(self.reservations.items()):
            if j is None:
                reserved[i] = 0
            elif j.state == JobState.RUNNING:
                end_time = j.start_time + j.walltime
                reserved[i] = end_time - current_time
            else:
                reserved[i] = j.walltime
        return reserved
