import os
import json

from ..network import *
from ..job import Job


def get_profile(data):
    if data['type'] == WorkloadProfileType.delay:
        return DelayProfile(data['delay'])
    elif data['type'] == WorkloadProfileType.parallel:
        return ParallelProfile(data['cpu'], data['com'])
    elif data['type'] == WorkloadProfileType.parallel_homogeneous:
        return ParallelHomogeneousProfile(data['cpu'], data['com'])
    elif data['type'] == WorkloadProfileType.parallel_homogeneous_total:
        return ParallelHomogeneousTotalProfile(data['cpu'], data['com'])
    else:
        raise NotImplementedError


class Workload():
    def __init__(self, name, fn):
        self.name = name
        self.path = fn
        with open(self.path, 'r') as f:
            data = json.load(f)
            self.simulation_time = data['simulation_time']
            self.profiles = {name: get_profile(
                profile) for name, profile in data['profiles'].items()}
            self.jobs = [Job(
                id="{}!{}".format(self.name, j['id']),
                res=j['res'],
                walltime=j['walltime'],
                profile=j['profile'],
                subtime=j['subtime'],
                user=j.get('user', "")) for j in data['jobs']]
            self.jobs.sort(key=lambda j: j.subtime)


class JobSubmitter(SimulationEventHandler):
    def __init__(self, simulator):
        super().__init__(simulator)
        self.current_workload = None
        self.workloads = []
        self._workload_finish_time = -1
        self.finished = True

    def _load_workload(self):
        assert self.workloads

        w = self.workloads.pop(0)
        w_name = "{}".format(w[w.rfind('/')+1:w.rfind('.json')])
        workload = Workload(w_name, w)

        for profile_name, profile in workload.profiles.items():
            self.simulator.register_profile(
                workload_name=workload.name,
                profile_name=profile_name,
                profile=profile
            )
        if self._workload_finish_time == -1:
            self._workload_finish_time = workload.simulation_time
        else:
            start_time = self._workload_finish_time
            for j in workload.jobs:
                j.subtime += start_time
            self._workload_finish_time += workload.simulation_time
        return workload

    def start(self, workloads):
        self.finished = False
        self._workload_finish_time = -1
        self.workloads = workloads.copy() if isinstance(
            workloads, list) else [workloads]
        self.current_workload = self._load_workload()
        self.simulator.call_me_later(self.current_workload.jobs[0].subtime)

    def close(self):
        self.finished = True
        self.current_workload = None
        self.workloads = []

    def on_requested_call(self, timestamp, data):
        if self.finished:
            return

        if timestamp != self._workload_finish_time and (len(self.current_workload.jobs) == 0 or timestamp != self.current_workload.jobs[0].subtime):
            return

        while self.current_workload.jobs and timestamp == self.current_workload.jobs[0].subtime:
            job = self.current_workload.jobs.pop(0)
            self.simulator.register_job(
                job.id,
                job.profile,
                job.res,
                job.walltime,
                job.user
            )

        if len(self.current_workload.jobs) == 0:
            if len(self.workloads) == 0 and timestamp == self._workload_finish_time:
                self.close()
                self.simulator.notify(NotifyType.no_more_static_job_to_submit)
            elif timestamp < self._workload_finish_time:
                self.simulator.call_me_later(self._workload_finish_time)
            else:
                self.current_workload = self._load_workload()
                self.simulator.call_me_later(
                    self.current_workload.jobs[0].subtime)
        else:
            self.simulator.call_me_later(self.current_workload.jobs[0].subtime)
