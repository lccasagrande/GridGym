import os
import json

from .network import *


class Workload():
    def __init__(self, name, fn):
        self.name = name
        with open(self.fn, 'r') as f:
            self.data = json.load(f)

    @property
    def profiles(self):
        return self.data['profiles']

    @property
    def jobs(self):
        return self.data['jobs']


class JobSubmitter():
    def __init__(self, workloads, protocol_handler, simulation_time):
        assert all(isinstance(w, Workload) for w in workloads)
        self.workloads = workloads
        # for w in os.listdir(workloads_dir) if w.endswith('.json')]
        self.simulation_time = simulation_time
        self.__protocol_handler = protocol_handler
        self.__protocol_handler.set_callback(
            EventType.REQUESTED_CALL, self._submitter)

    def start(self):
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

        for workload in self.workloads:
            for profile_name, profile in workload.profiles.items():
                self.__protocol_handler.register_profile(
                    workload_name=workload.name,
                    profile_name=profile_name,
                    profile=get_profile(profile)
                )

    def load_workload(self):
        pass

    def _submitter(self, _):
        pass
