import itertools
import copy
import json
from enum import Enum, auto
from collections import deque, defaultdict

import numpy as np
from sortedcontainers import SortedList


class JobState(Enum):
    NOT_SUBMITTED = 0
    SUBMITTED = 1
    RUNNING = 2
    COMPLETED_SUCCESSFULLY = 3
    COMPLETED_FAILED = 4
    COMPLETED_WALLTIME_REACHED = 5
    COMPLETED_KILLED = 6
    REJECTED = 7

    def __str__(self):
        return "%s" % self.__repr__()

    def __repr__(self):
        return self.name

    def __eq__(self, value):
        if isinstance(value, str):
            return self.name == value
        return super().__eq__(value)


class Job():
    def __init__(self, id, res, walltime, profile, subtime, user=""):
        self.id = id
        self.walltime = walltime
        self.res = res
        self.profile = profile
        self.subtime = subtime
        self.state = JobState.SUBMITTED
        self.user = user

        self.start_time = -1.  # will be set on start
        self.stop_time = -1.  # will be set on terminate
        self.allocation = None  # will be set on scheduling
        self.expected_time_to_start = -1.  # will be set on scheduling

    def __repr__(self):
        return "Job_%s" % self.id

    @property
    def workload_name(self):
        return self.id[:self.id.find("!")]

    @property
    def stretch(self):
        return -1 if self.start_time == -1 else self.waiting_time / self.walltime

    @property
    def waiting_time(self):
        return -1 if self.start_time == -1 else self.start_time - self.subtime

    @property
    def runtime(self):
        return -1 if self.stop_time == -1 else self.stop_time - self.start_time

    @property
    def turnaround_time(self):
        return -1 if self.stop_time == -1 else self.waiting_time + self.runtime

    @property
    def slowdown(self):
        return -1 if self.stop_time == -1 else self.turnaround_time / self.runtime

    def set_allocation(self, allocation):
        assert isinstance(allocation, list)
        assert len(allocation) == self.res
        self.allocation = allocation

    def start(self, current_time):
        assert self.allocation
        self.start_time = current_time
        self.state = JobState.RUNNING

    def terminate(self, current_time, state):
        assert isinstance(state, JobState)
        self.stop_time = current_time
        self.state = state

    def __eq__(self, value):
        return value.id == self.id if isinstance(value, self.__class__) else False
