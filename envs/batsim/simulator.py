import json
import numpy as np
from sortedcontainers import SortedList
from .scheduler import Job
from .network import BatsimEvent


class GridSimulator:
    def __init__(self, workloads, jobs_manager):
        self.jobs_manager = jobs_manager
        self.workloads = self._load_workloads(workloads)
        self.workload_idx = 0
        self.workload_nb_jobs = -1
        self.max_tracking_time_since_last_job = 10
        self.close()

    def close(self):
        self.curr_workload = None
        self.curr_workload_name = None
        self.workload_nb_jobs = -1
        self.jobs_submmited = -1
        self.jobs_completed = -1
        self.running = False
        self.current_time = -1
        self.time_since_last_new_job = -1

    def _load_workloads(self, workloads):
        def get_jobs(fn):
            with open(fn, 'r') as f:
                data = json.load(f)
                jobs = SortedList(key=lambda f: f.submit_time)
                for j in data['jobs']:
                    jobs.add(Job.from_json(j))
            return jobs

        return [(get_jobs(w), w) for w in workloads]

    def select_workload(self):
        if len(self.workloads) == self.workload_idx:
            self.workload_idx = 0
            np.random.shuffle(self.workloads)

        w = self.workloads[self.workload_idx]
        self.workload_idx += 1
        return w[0].copy(), w[1]

    def get_jobs_completed(self, time):
        jobs_running = self.jobs_manager.jobs_running
        for job in jobs_running:
            if job.remaining_time == 0:
                yield job

    def get_jobs_submmited(self, time):
        while len(self.curr_workload) > 0 and self.curr_workload[0].submit_time == time:
            yield self.curr_workload.pop(0)

    def reject_job(self, job_id):
        self.jobs_completed += 1

    def get_job_submitted_event(self, time, job):
        data = dict(job_id=job.id,
                    job=dict(profile=job.profile,
                             res=job.requested_resources,
                             id=job.id,
                             subtime=job.submit_time,
                             walltime=job.requested_time))
        return BatsimEvent(time, "JOB_SUBMITTED", data)

    def get_job_completed_event(self, time, job):
        data = dict(
            job_id=job.id,
            job_state=Job.State.COMPLETED,
            return_code=0,
            kill_reason="",
            alloc=job.allocation)
        return BatsimEvent(time, "JOB_COMPLETED", data)

    def get_simulation_ended_event(self, time):
        return BatsimEvent(time, "SIMULATION_ENDS", dict())

    def get_simulation_begins_event(self, time):
        return BatsimEvent(time, "SIMULATION_BEGINS", dict())

    @property
    def simulation_ended(self):
        return self.jobs_submmited == self.workload_nb_jobs and self.jobs_completed == self.workload_nb_jobs

    def proceed_time(self, t):
        self.current_time += t
        if self.time_since_last_new_job < self.max_tracking_time_since_last_job:
            self.time_since_last_new_job += t

    def start(self):
        self.curr_workload, self.curr_workload_name = self.select_workload()
        self.workload_nb_jobs = len(self.curr_workload)
        self.current_time = 0
        self.jobs_submmited = 0
        self.jobs_completed = 0
        self.time_since_last_new_job = 0
        self.running = True

    def read_events(self):
        assert self.running
        events = []

        for j in self.get_jobs_submmited(self.current_time):
            self.time_since_last_new_job = 0
            self.jobs_submmited += 1
            events.append(self.get_job_submitted_event(self.current_time, j))

        for j in self.get_jobs_completed(self.current_time):
            self.jobs_completed += 1
            events.append(self.get_job_completed_event(self.current_time, j))

        if self.simulation_ended:
            self.running = False
            events.append(self.get_simulation_ended_event(self.current_time))

        return events
