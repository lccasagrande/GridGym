import numpy as np
from sortedcontainers import SortedList
from collections import deque
from enum import Enum
import json


class JobSlot:
	def __init__(self, nb_slots):
		self.nb_slots = nb_slots
		self.slots = np.empty(shape=self.nb_slots, dtype=object)
		self.slots_free = SortedList(range(self.nb_slots))

	@property
	def is_empty(self):
		return len(self.slots_free) == self.nb_slots

	@property
	def is_full(self):
		return len(self.slots_free) == 0

	@property
	def jobs(self):
		return self.slots

	@property
	def nb_jobs(self):
		return self.nb_slots - len(self.slots_free)

	def clear(self):
		self.slots = np.empty(shape=self.nb_slots, dtype=object)
		self.slots_free = SortedList(range(self.nb_slots))

	def append(self, job):
		slot = self.slots_free.pop(0)
		self.slots[slot] = job

	def remove_at(self, slot):
		job = self.slots[slot]
		self.slots[slot] = None
		self.slots_free.add(value=slot)
		return job

	def at(self, slot):
		if slot >= len(self.slots):
			return None

		return self.slots[slot]


class JobManager:
	def __init__(self, nb_job_slots):
		self._job_slots = JobSlot(nb_job_slots)
		self._jobs_queue = deque()
		self._jobs_running = dict()
		self._jobs_allocated = dict()
		self.profiles = None
		self.first_job, self.last_job = None, None
		self.nb_jobs_submitted, self.nb_jobs_finished, self.nb_jobs_killed = 0, 0, 0
		self.total_waiting_time, self.total_slowdown, self.total_turnaround_time = 0, 0, 0
		self.max_waiting_time, self.max_turnaround_time, self.max_slowdown_time = 0, 0, 0
		self.runtime_slowdown, self.runtime_mean_slowdown = 0.0, 0.0

	@property
	def is_empty(self):
		return self._job_slots.is_empty

	@property
	def job_slots(self):
		return self._job_slots.jobs

	@property
	def nb_jobs_success(self):
		return self.nb_jobs_finished - self.nb_jobs_killed

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
	def nb_jobs_in_queue(self):
		return len(self._jobs_queue)

	@property
	def jobs_running(self):
		return list(self._jobs_running.values())

	@property
	def jobs_queue(self):
		return list(self._jobs_queue)

	def get_profile(self, profile_name):
		assert self.profiles is not None and profile_name in self.profiles
		profile = self.profiles[profile_name]
		return float(profile['cpu']), float(profile['com']), profile['type']

	def load(self, workload_fn):
		with open(workload_fn, 'r') as f:
			data = json.load(f)
			profiles = data['profiles']
		self.profiles = profiles

	def reset(self):
		self._job_slots.clear()
		self._jobs_queue.clear()
		self._jobs_running.clear()
		self._jobs_allocated.clear()
		self.first_job, self.last_job = None, None
		self.nb_jobs_submitted, self.nb_jobs_finished, self.nb_jobs_killed = 0, 0, 0
		self.total_waiting_time, self.total_slowdown, self.total_turnaround_time = 0, 0, 0
		self.max_waiting_time, self.max_turnaround_time, self.max_slowdown_time = 0, 0, 0
		self.runtime_slowdown, self.runtime_mean_slowdown = 0.0, 0.0

	def update_state(self, time_passed):
		for job in self._jobs_running.values():
			job.update_state(time_passed)
			self.runtime_slowdown += time_passed / job.requested_time

		for job in self._jobs_allocated.values():
			job.update_state(time_passed)
			self.runtime_slowdown += time_passed / job.requested_time

		for job in self._job_slots.jobs:
			if job is not None:
				job.update_state(time_passed)
				self.runtime_slowdown += time_passed / job.requested_time

		for job in self._jobs_queue:
			job.update_state(time_passed)
			self.runtime_slowdown += time_passed / job.requested_time

		self.runtime_mean_slowdown = self.runtime_slowdown / float(max(1, self.nb_jobs_submitted))

	def get_job_and_throw(self, index):
		job = self._job_slots.at(index)
		if job is None:
			raise InvalidJobError("There is no job at this position to schedule")
		return job

	def on_job_allocated(self, job_slot):
		job = self._job_slots.remove_at(job_slot)
		self._jobs_allocated[job.id] = job

		if self._jobs_queue:
			self._job_slots.append(self._jobs_queue.popleft())

	def on_job_started(self, job_id, time):
		job = self._jobs_allocated.pop(job_id)
		job.state = Job.State.RUNNING
		job.start_time = time
		job.time_left_to_start = 0
		self._jobs_running[job.id] = job
		if self.first_job is None:
			self.first_job = job

	def on_job_completed(self, job_id, time):
		job = self._jobs_running.pop(job_id)
		if job.expected_exec_time > job.runtime:
			job.state = Job.State.KILLED
			self.nb_jobs_killed += 1
		else:
			job.state = job.State.COMPLETED
		job.finish_time = time
		job.runtime = job.finish_time - job.start_time
		job.turnaround_time = job.waiting_time + job.runtime
		job.slowdown = job.turnaround_time / job.runtime
		self.last_job = job
		self._update_stats(job)
		self.nb_jobs_finished += 1
		return job

	def on_job_submitted(self, job, time):
		job.state = Job.State.SUBMITTED

		if self._job_slots.is_full:
			self._jobs_queue.append(job)
		else:
			self._job_slots.append(job)

		self.nb_jobs_submitted += 1

	def _update_stats(self, job):
		if job.waiting_time > self.max_waiting_time:
			self.max_waiting_time = job.waiting_time
		if job.turnaround_time > self.max_turnaround_time:
			self.max_turnaround_time = job.turnaround_time
		if job.slowdown > self.max_slowdown_time:
			self.max_slowdown_time = job.slowdown

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
		KILLED = 5

	def __init__(self, id, subtime, walltime, res, profile, cpu, type):
		self.id = id
		self.submit_time = subtime
		self.requested_time = walltime
		self.requested_resources = res
		self.cpu = cpu
		self.type = type
		self.profile = profile

		self.start_time = -1.  # will be set on scheduling
		self.time_left_to_start = -1.  # will be set on scheduling
		self.expected_exec_time = -1.  # will be set on scheduling
		self.finish_time = -1.  # will be set on completion
		self.turnaround_time = -1.  # will be set on completiom

		self.waiting_time = 0.
		self.runtime = 0.
		self.slowdown = 0.
		self.runtime_slowdown = 0.

		self.state = Job.State.NOT_SUBMITTED
		self.allocation = list()
		self.color = None

	@property
	def remaining_time(self):
		return max(0., self.requested_time - self.runtime)

	def update_state(self, time_passed):
		if self.state == Job.State.RUNNING:
			self.runtime += time_passed
		elif self.state == Job.State.SUBMITTED:
			self.waiting_time += time_passed

		if self.time_left_to_start != 0:
			self.time_left_to_start = max(0., self.time_left_to_start - time_passed)

		self.runtime_slowdown = (self.waiting_time + self.runtime) / self.requested_time


class InvalidJobError(Exception):
	pass
