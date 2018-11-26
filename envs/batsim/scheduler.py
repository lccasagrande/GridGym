import itertools
import math
import numpy as np
import random
from sortedcontainers import SortedList
from collections import deque
from matplotlib.colors import XKCD_COLORS as allcolors
from enum import Enum
from itertools import count
import heapq


class JobSlot():
	def __init__(self, slots):
		self.slots = np.empty(shape=slots, dtype=object)
		self.slots_free = SortedList(range(slots))

	@property
	def is_empty(self):
		return len(self.slots_free) == self.slots.shape[0]

	@property
	def values(self):
		return self.slots

	@property
	def is_full(self):
		return len(self.slots_free) == 0

	@property
	def nb_jobs(self):
		return self.slots.shape[0] - len(self.slots_free)

	def clear(self):
		self.slots = np.empty(shape=self.slots.shape, dtype=object)
		self.slots_free = SortedList(range(self.slots.shape[0]))

	def append(self, value):
		slot = self.slots_free.pop(0)
		self.slots[slot] = value

	def remove_at(self, slot):
		job = self.slots[slot]
		self.slots[slot] = None
		self.slots_free.add(slot)
		return job

	def at(self, slot):
		if slot >= len(self.slots):
			return None

		return self.slots[slot]


class SchedulerManager():
	def __init__(self, nb_resources, job_slots):
		self._job_slots = JobSlot(job_slots)
		self._jobs_queue = deque()
		self._jobs_running = dict()
		self._jobs_allocated = dict()
		self.reset()

	@property
	def is_empty(self):
		return self._job_slots.is_empty

	@property
	def job_slots(self):
		return self._job_slots.values

	@property
	def nb_jobs_in_slots(self):
		return self._job_slots.nb_jobs

	@property
	def nb_jobs_running(self):
		return len(self._jobs_running)

	@property
	def nb_jobs_allocated(self):
		return len(self._jobs_allocated)

	@property
	def nb_jobs_in_backlog(self):
		return len(self._jobs_queue)

	@property
	def jobs_running(self):
		return list(self._jobs_running.values())

	@property
	def jobs_queue(self):
		return list(self._jobs_queue)

	def lookup(self, index):
		return self._job_slots.at(index)

	def reset(self):
		self._job_slots.clear()
		self._jobs_queue.clear()
		self._jobs_running.clear()
		self._jobs_allocated.clear()
		self.first_job = None
		self.last_job = None
		self.nb_jobs_submitted = 0
		self.nb_jobs_completed = 0
		self.total_waiting_time = 0
		self.total_slowdown = 0
		self.total_turnaround_time = 0
		self.runtime_slowdown = 0.0
		self.runtime_mean_slowdown = 0.0

	def update_state(self, time_passed):
		for _, job in self._jobs_running.items():
			job.update_state(time_passed)
			self.runtime_slowdown += time_passed / job.requested_time

		for _, job in self._jobs_allocated.items():
			job.update_state(time_passed)
			self.runtime_slowdown += time_passed / job.requested_time

		for job in self._job_slots.values:
			if job != None:
				job.update_state(time_passed)
				self.runtime_slowdown += time_passed / job.requested_time

		for job in self._jobs_queue:
			job.update_state(time_passed)
			self.runtime_slowdown += time_passed / job.requested_time

		self.runtime_mean_slowdown = self.runtime_slowdown / max(1.0, float(self.nb_jobs_submitted))

	def get_job(self, index):
		job = self._job_slots.at(index)
		if job is None:
			raise InvalidJobError(
				"There is no job at this position to schedule")
		return job

	def get_max_slowdown(self):
		slowdown = 0
		for j in self.job_slots:
			if j is not None:
				j_s = j.estimate_slowdown()
				if j_s > slowdown:
					slowdown = j_s
		return slowdown

	def get_max_waiting_time(self):
		waiting_time = 0
		for j in self.job_slots:
			if j is not None and j.waiting_time > waiting_time:
				waiting_time = j.waiting_time
		return waiting_time

	def on_job_allocated(self, index):
		job = self._job_slots.remove_at(index)
		self._jobs_allocated[job.id] = job

		if self._jobs_queue:
			job = self._jobs_queue.popleft()
			self._job_slots.append(job)

	def on_job_started(self, job_id, time):
		job = self._jobs_allocated.pop(job_id)
		job.state = Job.State.RUNNING
		job.start_time = time
		job.time_left_to_start = 0
		self._jobs_running[job.id] = job
		if self.first_job == None:
			self.first_job = job

	def on_job_completed(self, time, data):
		job = self._jobs_running.pop(data['job_id'])
		job.finish_time = time
		job.runtime = job.finish_time - job.start_time
		job.turnaround_time = job.waiting_time + job.runtime
		job.slowdown = job.turnaround_time / job.runtime
		job.state = Job.State.COMPLETED
		assert job.remaining_time == 0

		self._update_stats(job)
		self.last_job = job
		self.nb_jobs_completed += 1
		return job

	def on_job_submitted(self, time, data):
		job = Job.from_json(data)
		job.state = Job.State.SUBMITTED

		if self._job_slots.is_full:
			self._jobs_queue.append(job)
		else:
			self._job_slots.append(job)

		self.nb_jobs_submitted += 1

	def _update_stats(self, job):
		self.total_slowdown += job.slowdown
		self.total_waiting_time += job.waiting_time
		self.total_turnaround_time += job.turnaround_time


class Job(object):
	class State(Enum):
		NOT_SUBMITTED = 0
		SUBMITTED = 1
		RUNNING = 2
		COMPLETED = 3
		REJECTED = 4

	def __init__(
			self,
			id,
			subtime,
			walltime,
			res,
			profile):
		self.id = id
		self.submit_time = subtime
		self.requested_time = walltime
		self.requested_resources = res
		self.profile = profile
		self.start_time = -1.  # will be set on scheduling by batsim
		self.time_left_to_start = -1.  # will be set on scheduling by batsim
		self.finish_time = -1.  # will be set on completion by batsim
		self.turnaround_time = -1.  # will be set on completion by batsim
		self.waiting_time = 0.  # will be set on completion by batsim
		self.runtime = 0.  # will be set on completion by batsim
		self.slowdown = 0.
		self.runtime_slowdown = 0.
		self.state = Job.State.NOT_SUBMITTED
		self.allocation = []
		self.color = None

	@property
	def remaining_time(self):
		return self.requested_time - self.runtime if self.finish_time == -1 else 0.0

	def update_state(self, time_passed):
		if self.state == Job.State.RUNNING:
			self.runtime += time_passed
		elif self.state == Job.State.SUBMITTED:
			self.waiting_time += time_passed

		if self.time_left_to_start > 0:
			self.time_left_to_start = max(0, math.ceil(self.time_left_to_start - time_passed))

		runtime_turnaround = self.waiting_time + self.runtime
		self.runtime_slowdown = runtime_turnaround / self.requested_time

	def estimate_slowdown(self):
		return ((self.requested_time + self.waiting_time) / self.requested_time) - 1

	@staticmethod
	def from_json(json_dict):
		return Job(json_dict["id"],
		           json_dict["subtime"],
		           json_dict.get("walltime", -1),
		           json_dict["res"],
		           json_dict["profile"])


class InsufficientResourcesError(Exception):
	pass


class InvalidJobError(Exception):
	pass
