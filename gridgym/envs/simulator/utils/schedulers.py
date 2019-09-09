import math
from copy import copy
from abc import ABC, abstractmethod

import numpy as np


class Scheduler(ABC):
    @abstractmethod
    def schedule(self, queue, reserved_time):
        raise NotImplementedError()


class FirstComeFirstServed(Scheduler):
    def schedule(self, queue, reserved_time):
        jobs = []
        if len(queue) > 0:
            available = [i for i, a in enumerate(reserved_time) if a == 0]
            for job in queue:
                if job.res <= len(available):
                    reservation = available[:job.res]
                    reserved_time[reservation] = job.walltime
                    job.expected_time_to_start = 0
                    jobs.append(job.id)
                    del available[:job.res]
                else:
                    break
        return jobs


class EASYBackfilling(Scheduler):
    def __init__(self):
        super().__init__()
        self.fcfs = FirstComeFirstServed()
        self._priority_job = None

    def _backfill(self, queue, nb_available_resources):
        assert self._priority_job
        jobs = []
        for job in queue:
            if nb_available_resources == 0:
                break
            elif job.res <= nb_available_resources and job.walltime <= self._priority_job.expected_time_to_start:
                jobs.append(job.id)
                job.expected_time_to_start = 0
                nb_available_resources -= job.res
        return jobs

    def schedule(self, queue, reserved_time):
        self._priority_job = None
        jobs = self.fcfs.schedule(queue, reserved_time)
        if len(jobs) < len(queue):  # There is some jobs that could not be scheduled
            # Let's give an upper bound start time for the first job in the queue
            self._priority_job = queue[len(jobs)]
            earliest_time = sorted(reserved_time)[:self._priority_job.res]
            earliest_time = math.ceil(
                earliest_time[-1]) if earliest_time else 0
            self._priority_job.expected_time_to_start = earliest_time
            queue = queue[len(jobs) + 1:]
            if len(queue) > 0 and self._priority_job.expected_time_to_start > 0:
                nb_available = np.count_nonzero(reserved_time == 0)
                backfill_jobs = self._backfill(queue, nb_available)
                jobs.extend(backfill_jobs)
        return jobs


class SAFBackfilling(EASYBackfilling):
    def _backfill(self, queue, nb_available_resources):
        saf_queue = sorted(queue, key=lambda j: j.walltime * j.res)
        return super()._backfill(saf_queue, nb_available_resources)
