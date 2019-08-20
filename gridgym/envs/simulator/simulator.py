import subprocess
import time as tm
import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod

from sortedcontainers import SortedDict

from .resource import Platform, PowerStateType, ResourceState
from .job import Job, JobState
from .network import ProtocolHandler, EventType, NotifyType
from .submitter import JobSubmitter


class SimulationHandler():
    def __init__(self):
        self.__protocol_handler = ProtocolHandler()
        self.__protocol_handler.set_callback(
            EventType.JOB_COMPLETED, self._handle_job_completed)
        self.__protocol_handler.set_callback(
            EventType.JOB_COMPLETED, self.__start_ready_jobs)
        self.__protocol_handler.set_callback(
            EventType.JOB_SUBMITTED, self._handle_job_submitted)
        self.__protocol_handler.set_callback(
            EventType.SIMULATION_BEGINS, self._handle_simulation_begins)
        self.__protocol_handler.set_callback(
            EventType.SIMULATION_ENDS, self._handle_simulation_ends)
        self.__protocol_handler.set_callback(
            EventType.JOB_KILLED, self._handle_job_killed)
        self.__protocol_handler.set_callback(
            EventType.REQUESTED_CALL, self._handle_requested_call)
        self.__protocol_handler.set_callback(
            EventType.RESOURCE_STATE_CHANGED, self._handle_resource_state_changed)
        self.__protocol_handler.set_callback(
            EventType.RESOURCE_STATE_CHANGED, self.__start_ready_jobs)
        self.__protocol_handler.set_callback(
            EventType.NOTIFY, self._handle_notify)
        self.__protocol_handler.set_callback(
            EventType.JOB_KILLED, self.__start_ready_jobs)
        self.__jobs = {"queue": [], "running": {},
                       "completed": [], "ready": []}
        self.__simulator = None
        self.platform, self.simulation_time, self.submitter_ended = None, None, False

    @property
    def current_time(self):
        return self.__protocol_handler.current_time

    @property
    def is_running(self):
        return self.__simulator is not None and (not self.submitter_ended or any(v for k, v in self.__jobs.items() if k != 'completed'))

    @property
    def jobs_queue(self):
        return np.array(self.__jobs["queue"])

    @property
    def jobs_running(self):
        return np.asarray(list(self.__jobs["running"].values()))

    @property
    def jobs_completed(self):
        return np.array(self.__jobs["completed"])

    @property
    def agenda(self):
        return np.asarray([
            (r.allocated_job.start_time + r.allocated_job.walltime) - self.current_time if r.is_computing
            else r.allocated_job.walltime if r.is_reserved
            else 0
            for r in self.platform.resources
        ])

    def start(self, workload_fn, platform_fn, output_dir=None, simulation_time=None):
        assert not self.is_running, "A simulation is already running."
        assert not simulation_time or simulation_time > 0

        cmd = "batsim -s {} -p {} -w {} -E --enable-dynamic-jobs -q".format(
            self.__protocol_handler.address, platform_fn, workload_fn
        )
        cmd += " -e {}".format(output_dir) if output_dir else ""

        self.__simulator = subprocess.Popen(
            cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, shell=False
        )

        self.simulation_time = simulation_time
        self.submitter_ended = False
        self.__protocol_handler.start()

    def close(self):
        self.__protocol_handler.close()
        if self.__simulator:
            self.__simulator.terminate()
        self.__simulator = None

    def proceed_time(self, until=0):
        assert self.__simulator, "Cannot proceed if there is no simulator running."
        self.__start_ready_jobs()
        if until > 0:
            assert until > self.current_time
            self.__protocol_handler.call_me_later(until + 0.001)
            while self.is_running and self.current_time < until:
                self.__protocol_handler.proceed_simulation()
        else:
            self.__protocol_handler.proceed_simulation()

        self.check_end_of_simulation()

    def check_end_of_simulation(self):
        if not self.is_running and self.__simulator is not None:
            self.__protocol_handler.notify(NotifyType.registration_finished)
            while self.__simulator:
                self.__protocol_handler.proceed_simulation()

    def _handle_simulation_begins(self, data):
        self.platform = data.platform
        if self.simulation_time:
            self.__protocol_handler.call_me_later(self.simulation_time)

    def _handle_simulation_ends(self, _):
        self.__protocol_handler.ack()
        self.__simulator.wait()
        self.__simulator.terminate()
        self.__simulator = None
        self.__protocol_handler.close()

    def _handle_job_submitted(self, data):
        self.__jobs["queue"].append(data.job)

    def _handle_job_completed(self, data):
        job = self.__jobs["running"].pop(data.job_id)
        job.terminate(self.current_time, data.job_state)
        self.__jobs["completed"].append(job)
        for r in self.platform.get_resources(job.allocation):
            r.release()

    def _handle_job_killed(self, data):
        for id in data.job_ids:
            job = self.__jobs["running"].pop(data.job_id)
            job.terminate(self.current_time, JobState.COMPLETED_KILLED)
            self.__jobs["completed"].append(job)
            self.platform.release(job.alloc)

    def _handle_requested_call(self, data):
        if self.simulation_time and self.current_time >= self.simulation_time:
            if self.is_running:
                self.close()
            else:
                self.check_end_of_simulation()

    def _handle_notify(self, data):
        if data.type == NotifyType.no_more_static_job_to_submit:
            self.submitter_ended = True

    def _handle_resource_state_changed(self, data):
        resources = self.platform.get_resources(data.resources)
        for r in resources:
            r.set_pstate(data.state)

    def set_pstate(self, *node_ids, pstate_id):
        resources_ids = []
        for i in node_ids:
            node = self.platform.get_node(i)
            node.set_pstate(pstate_id)
            resources_ids.extend([r.id for r in node.resources])

        self.__protocol_handler.set_resources_state(resources_ids, pstate_id)

    def turn_on(self, *node_ids):
        resources_new_state = defaultdict(list)
        for i in node_ids:
            node = self.platform.get_node(i)
            node.wakeup()
            ps_id = next(
                ps.id for ps in node.power_states if ps.type == PowerStateType.computation)
            resources_new_state[ps_id].extend([r.id for r in node.resources])

        for ps_id, resources_ids in resources_new_state.items():
            self.__protocol_handler.set_resources_state(resources_ids, ps_id)

    def turn_off(self, *node_ids):
        resources_new_state = defaultdict(list)
        for i in node_ids:
            node = self.platform.get_node(i)
            node.sleep()
            ps_id = next(
                ps.id for ps in node.power_states if ps.type == PowerStateType.sleep)
            resources_new_state[ps_id].extend([r.id for r in node.resources])

        for ps_id, resources_ids in resources_new_state.items():
            self.__protocol_handler.set_resources_state(resources_ids, ps_id)

    def initiate_resources(self, resources):
        ready, nodes_visited = True, {}
        for r in resources:
            if not r.is_idle and r.parent_id not in nodes_visited:
                ready = False
                nodes_visited[r.parent_id] = True
                if r.is_sleeping:
                    self.turn_on(r.parent_id)
        return ready

    def __start_ready_jobs(self, _=None):
        for job in list(self.__jobs['ready']):
            allocated_resources = self.platform.get_resources(job.allocation)
            ready = self.initiate_resources(allocated_resources)
            if ready:
                for r in allocated_resources:
                    r.start_computing()
                job.start(self.current_time)
                self.__protocol_handler.execute_job(job.id, job.allocation)
                self.__jobs['ready'].remove(job)
                self.__jobs['running'][job.id] = job

    def allocate(self, job_id, resource_ids=None):
        job_idx = next(i for i, j in enumerate(
            self.__jobs['queue']) if j.id == job_id)
        job = self.__jobs['queue'][job_idx]
        if not resource_ids:
            allocation = sorted(
                [r for r in self.platform.resources if not r.is_reserved], key=lambda r: r.state.value)[:job.res]
            resource_ids = [r.id for r in allocation]
        else:
            allocation = self.platform.get_resources(resource_ids)

        if len(allocation) < job.res:
            raise Exception("Insufficient resources for job {}".format(job.id))

        del self.__jobs['queue'][job_idx]
        for r in allocation:
            r.reserve(job)
        job.set_allocation(resource_ids)
        self.__jobs['ready'].append(job)
