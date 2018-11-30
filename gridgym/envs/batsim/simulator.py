import json
import numpy as np
import os
import pandas as pd
import subprocess
import math
from sortedcontainers import SortedList
from .scheduler import Job, SchedulerManager
from .network import BatsimEvent, BatsimProtocolHandler
from .resource import Resource, ResourceManager


class GridSimulator:
	def __init__(self, workloads, jobs_manager):
		self.jobs_manager = jobs_manager
		self.workloads = self._load_workloads(workloads)
		self.workload_idx = 0
		self.workload_nb_jobs = -1
		self.close()

	def close(self):
		self.curr_workload = None
		self.curr_workload_name = None
		self.workload_nb_jobs = -1
		self.jobs_submmited = -1
		self.jobs_completed = -1
		self.running = False
		self.current_time = -1

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


class SimulatorHandler:
	PLATFORM = "platforms/platform_hg_500.xml"
	WORKLOAD_DIR = "workloads"
	OUTPUT_DIR = "../results"

	def __init__(self, job_slots, time_window, time_slice, backlog_width):
		assert time_window % time_slice == 0
		self.files_path = os.path.join(os.path.dirname(__file__), "files")
		if not os.path.exists(self.files_path):
			raise IOError("File %s does not exist" % self.files_path)

		self.time_window = time_window
		self.time_slice = time_slice
		self.state_time_window = int(self.time_window / self.time_slice)
		self.job_slots = job_slots
		self.backlog_width = backlog_width
		self.nb_simulation = 0
		self.max_tracking_time_since_last_job = 10

		os.makedirs(SimulatorHandler.OUTPUT_DIR, exist_ok=True)
		self._platform = os.path.join(self.files_path, SimulatorHandler.PLATFORM)
		workloads_path = os.path.join(self.files_path, SimulatorHandler.WORKLOAD_DIR)
		self._workloads = [workloads_path + "/" + w for w in os.listdir(workloads_path) if w.endswith('.json')]
		self.resource_manager = ResourceManager.from_xml(self._platform, self.time_window)
		self.jobs_manager = SchedulerManager(self.job_slots)
		self._init_vars()

	@property
	def nb_jobs_waiting(self):
		return self.jobs_manager.nb_jobs_in_backlog + self.jobs_manager.nb_jobs_in_slots

	@property
	def nb_jobs_running(self):
		return self.jobs_manager.nb_jobs_running

	@property
	def nb_jobs_submitted(self):
		return self.jobs_manager.nb_jobs_submitted

	@property
	def nb_jobs_completed(self):
		return self.jobs_manager.nb_jobs_completed

	@property
	def nb_resources(self):
		return self.resource_manager.nb_resources

	@property
	def current_time(self):
		raise NotImplementedError()

	@property
	def is_running(self):
		raise NotImplementedError()

	def schedule(self, job_pos):
		job = self.jobs_manager.get_job(job_pos)
		self.resource_manager.allocate(job)
		self.jobs_manager.on_job_allocated(job_pos)

	def _get_ready_jobs(self):
		ready_jobs = []
		jobs = self.resource_manager.get_jobs()
		for job in jobs:
			if job is not None and \
					job.state != Job.State.RUNNING and \
					job.time_left_to_start == 0 and \
					job not in ready_jobs and \
					self.resource_manager.is_available(job.allocation):
				ready_jobs.append(job)
		return ready_jobs

	def _start_ready_jobs(self):
		jobs = self.resource_manager.get_jobs()
		for job in jobs:
			if job.state != Job.State.RUNNING and \
					job.time_left_to_start == 0 and \
					self.resource_manager.is_available(job.allocation):
				self._start_job(job)

	def _handle_job_completed(self, timestamp, data):
		job = self.jobs_manager.on_job_completed(timestamp, data)
		self.resource_manager.release(job)
		self._start_ready_jobs()

	def _handle_job_submitted(self, timestamp, data):
		if data['job']['res'] <= self.resource_manager.nb_resources:
			self.jobs_manager.on_job_submitted(timestamp, data['job'])
			self.time_since_last_new_job = timestamp
		else:
			self._reject_job(data['job_id'])

	def _handle_simulation_ends(self, data):
		self.metrics = data
		self._export_metrics()

	def _handle_requested_call(self, timestamp):
		pass

	def _handle_simulation_begins(self, data):
		pass

	def _handle_event(self, event):
		if event.type == "SIMULATION_BEGINS":
			self._handle_simulation_begins(event.data)
		if event.type == "SIMULATION_ENDS":
			self._handle_simulation_ends(event.data)
		elif event.type == "JOB_SUBMITTED":
			self._handle_job_submitted(event.timestamp, event.data)
		elif event.type == "JOB_COMPLETED":
			self._handle_job_completed(event.timestamp, event.data)
		elif event.type == "REQUESTED_CALL":
			self._handle_requested_call(event.timestamp)

	def get_job_slot_state(self):
		s = np.zeros(shape=(self.state_time_window, 2 * self.job_slots), dtype=np.float)
		#
		# for i, job in enumerate(self.jobs_manager.job_slots):
		#	if job is not None:
		#		start_idx = i * self.nb_resources
		#		end_idx = start_idx + job.requested_resources
		#		s[0:job.requested_time, start_idx:end_idx] = 1
		i = 0
		for j in self.jobs_manager.job_slots:
			if j is not None:
				frac, whole = math.modf(j.requested_time / self.time_slice)
				job_time = [1.0 for _ in range(int(whole))]
				if frac != 0:
					job_time.append(frac)
				req_res = j.requested_resources / float(self.nb_resources)
				lenght = len(job_time)
				job_res = [req_res for _ in range(lenght)]

				s[0:lenght, i] = job_res
				s[0:lenght, i + 1] = job_time
			i += 2
		return s

	def get_backlog_state(self):
		s = np.zeros(shape=(self.state_time_window, self.backlog_width), dtype=np.uint8)
		t, i = 0, 0
		nb_jobs = min(self.backlog_width * self.state_time_window, self.jobs_manager.nb_jobs_in_backlog)
		for _ in range(nb_jobs):
			s[t, i] = 1
			i += 1
			if i == self.backlog_width:
				i = 0
				t += 1
		return s

	def get_time_state(self):
		diff = min(self.max_tracking_time_since_last_job, self.current_time - self.time_since_last_new_job)
		v = diff / float(self.max_tracking_time_since_last_job)
		return np.full(shape=self.state_time_window, fill_value=v, dtype=np.float)

	def get_resource_state(self):
		res_state = self.resource_manager.get_view()
		res_state = np.array(
			[np.mean(res_state[t:t + self.time_slice, :], axis=0) for t in range(0, self.time_window, self.time_slice)])
		return res_state

	def get_easy_state(self):
		state = np.zeros(shape=(self.nb_resources + self.job_slots * 2))
		# RESOURCES
		state[0:self.nb_resources] = [res.is_computing for k, res in self.resource_manager.resources.items()]

		i = self.nb_resources
		for j in self.jobs_manager.job_slots:
			if j is not None:
				state[i] = j.requested_resources
				state[i + 1] = j.requested_time
			i += 2
		return state

	def get_compact_state(self):
		state = np.zeros(shape=(self.nb_resources + self.job_slots * 2 + 2))
		# RESOURCES
		state[0:self.nb_resources] = [res.get_reserved_time() / float(15.) for k, res in
		                              self.resource_manager.resources.items()]

		# JOB SLOTS
		start_idx = self.nb_resources
		for j in self.jobs_manager.job_slots:
			if j is not None:
				state[start_idx] = j.requested_resources / float(self.nb_resources)
				state[start_idx + 1] = j.requested_time / float(15.)
			start_idx += 2

		# BACKLOG
		max_backlog_jobs = self.backlog_width * self.time_window
		state[start_idx] = min(max_backlog_jobs, self.jobs_manager.nb_jobs_in_backlog) / float(max_backlog_jobs)

		# LAST TIME
		diff = min(self.max_tracking_time_since_last_job, self.current_time - self.time_since_last_new_job)
		state[start_idx + 1] = diff / float(self.max_tracking_time_since_last_job)

		return state

	def get_state(self):
		shape = (self.state_time_window, self.nb_resources + self.job_slots * 2 + self.backlog_width + 1)
		state = np.zeros(shape=shape, dtype=np.float)

		# RESOURCES
		resource_end = self.nb_resources
		state[:, 0:resource_end] = self.get_resource_state()

		# JOB SLOTS
		job_slot_end = 2 * self.job_slots + resource_end
		state[:, resource_end:job_slot_end] = self.get_job_slot_state()

		# BACKLOG
		backlog_end = job_slot_end + self.backlog_width
		state[:, job_slot_end:backlog_end] = self.get_backlog_state()

		# LAST SUB TIME
		state[:, -1] = self.get_time_state()

		return state

	def start(self):
		raise NotImplementedError()

	def close(self):
		raise NotImplementedError()

	def _start_job(self, job):
		raise NotImplementedError()

	def _reject_job(self, job_id):
		raise NotImplementedError()

	def _init_vars(self):
		self.metrics = {}
		self.time_since_last_new_job = 0
		self.jobs_manager.reset()
		self.resource_manager.reset()

	def _export_metrics(self):
		data = pd.DataFrame(self.metrics, index=[0])
		fn = "{}/env_{}_metrics.csv".format(SimulatorHandler.OUTPUT_DIR, self.nb_simulation)
		data.to_csv(fn, index=False)


class BatsimHandler(SimulatorHandler):
	USE_DOCKER = False

	def __init__(self, job_slots, time_window, time_slice, backlog_width, verbose='quiet'):
		self.protocol_manager = BatsimProtocolHandler()
		self.running_simulation = False
		self._simulator_process = None
		self.verbose = verbose
		self._workload_idx = 0
		super(BatsimHandler, self).__init__(job_slots, time_window, time_slice, backlog_width)

	@property
	def current_time(self):
		return self.protocol_manager.current_time

	@property
	def is_running(self):
		return self.running_simulation

	def schedule(self, job_pos):
		assert self.running_simulation, "Simulation is not running."
		if job_pos != -1:  # Try to schedule job
			super(BatsimHandler, self).schedule(job_pos)
			self._start_ready_jobs()
			return

		self._checkpoint()

		self._start_ready_jobs()

		self._wait_state_change()

	def close(self):
		self.protocol_manager.close()
		self.running_simulation = False
		if self._simulator_process is not None:
			self._simulator_process.terminate()
			self._simulator_process.wait()
			self._simulator_process = None

	def start(self):
		assert not self.is_running, "A simulation is already running."
		self.nb_simulation += 1
		self._init_vars()
		self._simulator_process = self._start_simulator()
		self.protocol_manager.start()
		while self.jobs_manager.is_empty:
			self._update_state()

		assert self.is_running, "An error ocurred during simulator starting."

	def _checkpoint(self):
		if not self._alarm_is_set:
			self.protocol_manager.set_alarm(self.current_time + 1)
			self._alarm_is_set = True

	def _wait_state_change(self):
		# slots_occuped = self.jobs_manager.nb_jobs_in_slots
		self._update_state()
		while self.running_simulation and (self.alarm_time != -1 or self.jobs_manager.is_empty):  # and self.jobs_manager.nb_jobs_in_slots == slots_occuped:
			#self._checkpoint()
			self._update_state()

	def _start_job(self, job):
		r = self.resource_manager.start_job(job)
		self.jobs_manager.on_job_started(job.id, self.current_time)
		if len(r) != 0:
			self.protocol_manager.set_resource_pstate(r, Resource.PowerState.NORMAL)
		self.protocol_manager.start_job(job.id, job.allocation)

	def _select_workload(self):
		if len(self._workloads) == self._workload_idx:
			self._workload_idx = 0
			np.random.shuffle(self._workloads)

		x = self._workloads[self._workload_idx]
		self._workload_idx += 1
		return x

	def _start_simulator(self):
		platform_file = self._platform
		workload_file = self._select_workload()

		if BatsimHandler.USE_DOCKER:
			cmd = "docker run --net host -v {}:/batsim lccasagrande/batsim batsim".format(self.files_path)
			platform_file = platform_file.replace(self.files_path + "/", "")
			workload_file = workload_file.replace(self.files_path + "/", "")
			output_fn = "results/bat_" + str(self.nb_simulation)
		else:
			cmd = "batsim"
			output_fn = SimulatorHandler.OUTPUT_DIR + "/bat_" + str(self.nb_simulation)

		cmd += " -s {} -p {} -w {} -v {} -E -e {}".format(
			self.protocol_manager.socket_endpoint, platform_file,
			workload_file, self.verbose, output_fn)
		return subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, shell=False)

	def _init_vars(self):
		super(BatsimHandler, self)._init_vars()
		self.alarm_time = -1
		self._alarm_is_set = False
		self.protocol_manager.reset()

	def _handle_requested_call(self, timestamp):
		self._alarm_is_set = False

	def _reject_job(self, job_id):
		self.protocol_manager.reject_job(job_id)

	def _handle_simulation_ends(self, data):
		assert self.is_running, "No simulation is currently running"
		self.protocol_manager.acknowledge()
		self.protocol_manager.send_events()
		self.running_simulation = False
		metrics = dict(
			makespan=float(data["makespan"]),
			mean_waiting_time=float(data["mean_waiting_time"]),
			mean_turnaround_time=float(data["mean_turnaround_time"]),
			mean_slowdown=float(data["mean_slowdown"]),
			max_waiting_time=float(data["max_waiting_time"]),
			max_turnaround_time=float(data["max_turnaround_time"]),
			max_slowdown=float(data["max_slowdown"]),
			energy_consumed=float(data["consumed_joules"]),
			total_slowdown=self.jobs_manager.total_slowdown,
			total_turnaround_time=self.jobs_manager.total_turnaround_time,
			total_waiting_time=self.jobs_manager.total_waiting_time
		)
		super(BatsimHandler, self)._handle_simulation_ends(metrics)

	def _handle_simulation_begins(self, data):
		assert not self.is_running, "A simulation is already running (is more than one instance of Batsim active?!)"
		self.running_simulation = True

	def _update_state(self):
		self.protocol_manager.send_events()

		old_time = self.current_time
		events = self.protocol_manager.read_events(blocking=not self.running_simulation)

		# New jobs does not need to be updated in this timestep
		# Update jobs if no time has passed does not make sense.
		time_passed = self.current_time - old_time
		if time_passed != 0:
			if self.alarm_time >= self.current_time:
				self.alarm_time = -1

			self.jobs_manager.update_state(time_passed)
			self.resource_manager.update_state(time_passed)

		for event in events:
			self._handle_event(event)

		# Remember to always ack
		if self.running_simulation:
			self.protocol_manager.acknowledge()


class GridSimulatorHandler(SimulatorHandler):
	def __init__(self, job_slots, time_window, time_slice, backlog_width):
		super(GridSimulatorHandler, self).__init__(job_slots, time_window, time_slice, backlog_width)
		self.simulation_manager = GridSimulator(self._workloads, self.jobs_manager)

	def schedule(self, job_pos):
		assert self.is_running, "Simulation is not running."

		if job_pos != -1:  # Try to schedule job
			super(GridSimulatorHandler, self).schedule(job_pos)
		else:
			self._proceed_time()

		self._start_ready_jobs()

		self._update_state()

	@property
	def current_time(self):
		return self.simulation_manager.current_time

	@property
	def is_running(self):
		return self.simulation_manager.running

	def start(self):
		assert not self.is_running, "A simulation is already running."
		self.simulation_manager.start()
		self._init_vars()
		self._wait_state_change()
		assert self.is_running, "An error ocurred during simulator starting."
		self.nb_simulation += 1

	def close(self):
		self.simulation_manager.close()

	def _start_job(self, job):
		self.resource_manager.start_job(job)
		self.jobs_manager.on_job_started(job.id, self.current_time)

	def _reject_job(self, job_id):
		self.simulation_manager.reject_job(job_id)

	def _wait_state_change(self):
		self._update_state()
		while self.is_running and (self.jobs_manager.is_empty or self.resource_manager.is_full()):
			self._proceed_time()
			self._update_state()

	def _proceed_time(self):
		self.simulation_manager.proceed_time(1)
		self.jobs_manager.update_state(1)
		self.resource_manager.update_state(1)

	def _update_state(self):
		events = self.simulation_manager.read_events()
		for event in events:
			self._handle_event(event)

	def _handle_simulation_ends(self, data):
		metrics = dict(
			makespan=self.jobs_manager.last_job.finish_time,
			mean_waiting_time=self.jobs_manager.total_waiting_time / self.jobs_manager.nb_jobs_submitted,
			mean_turnaround_time=self.jobs_manager.total_turnaround_time / self.jobs_manager.nb_jobs_submitted,
			mean_slowdown=self.jobs_manager.runtime_mean_slowdown,
			max_waiting_time=self.jobs_manager.max_waiting_time,
			max_turnaround_time=self.jobs_manager.max_turnaround_time,
			max_slowdown=self.jobs_manager.max_slowdown_time,
			energy_consumed=self.resource_manager.energy_consumption,
			total_slowdown=self.jobs_manager.total_slowdown,
			total_turnaround_time=self.jobs_manager.total_turnaround_time,
			total_waiting_time=self.jobs_manager.total_waiting_time
		)
		super(GridSimulatorHandler, self)._handle_simulation_ends(metrics)
