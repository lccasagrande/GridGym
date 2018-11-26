import os
import pandas as pd
import subprocess
import numpy as np
import math
from .resource import Resource, ResourceManager
from .scheduler import SchedulerManager, Job
from .network import BatsimProtocolHandler
from .simulator import GridSimulator


class BatsimHandler:
	PLATFORM = "platforms/platform_hg_10.xml"
	WORKLOAD_DIR = "workloads"
	OUTPUT_DIR = "results/batsim"

	def __init__(self, job_slots, time_window, backlog_width, verbose='quiet'):
		fullpath = os.path.join(os.path.dirname(__file__), "files")
		if not os.path.exists(fullpath):
			raise IOError("File %s does not exist" % fullpath)

		self._output_dir = self._make_random_dir(BatsimHandler.OUTPUT_DIR)
		self._platform = os.path.join(fullpath, BatsimHandler.PLATFORM)
		workloads_path = os.path.join(fullpath, BatsimHandler.WORKLOAD_DIR)
		self._workloads = [workloads_path + "/" +
		                   w for w in os.listdir(workloads_path) if w.endswith('.json')]
		self._workload_idx = 0
		self.max_tracking_time_since_last_job = 10
		self._simulator_process = None
		self.running_simulation = False
		self.nb_simulation = 0
		self._verbose = verbose
		self.time_window = time_window
		self.job_slots = job_slots
		self.protocol_manager = BatsimProtocolHandler()
		self.resource_manager = ResourceManager.from_xml(self._platform, self.time_window)
		self.jobs_manager = SchedulerManager(self.nb_resources, job_slots)
		self.backlog_width = backlog_width
		self.state_shape = (self.time_window, self.nb_resources + (self.nb_resources * self.job_slots) + backlog_width)
		self._reset()

	@property
	def nb_jobs_waiting(self):
		return self.jobs_manager.nb_jobs_in_backlog + self.jobs_manager.job_slots.lenght

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
		return self.protocol_manager.current_time

	def get_state(self):
		return self._get_image()

	def close(self):
		self.protocol_manager.close()
		self.running_simulation = False
		if self._simulator_process is not None:
			self._simulator_process.terminate()
			self._simulator_process.wait()
			self._simulator_process = None

	def start(self):
		assert not self.running_simulation, "A simulation is already running."
		self._reset()
		self._simulator_process = self._start_simulator()
		self.protocol_manager.start()
		while self.jobs_manager.is_empty:
			self._update_state()

		assert self.running_simulation, "An error ocurred during simulator starting."
		self.nb_simulation += 1

	def schedule(self, job_pos):
		assert self.running_simulation, "Simulation is not running."
		if job_pos != -1:  # Try to schedule job
			job = self.jobs_manager.get_job(job_pos)
			self.resource_manager.allocate(job)
			self.jobs_manager.on_job_allocated(job_pos)
			self._start_ready_jobs()
			return
		else:
			self.alarm_time = self.current_time + 1

		self._checkpoint()

		self._start_ready_jobs()

		self._wait_state_change()

	def _checkpoint(self):
		if not self._alarm_is_set:
			self.protocol_manager.set_alarm(self.current_time + 1)
			self._alarm_is_set = True

	def _wait_state_change(self):
		slots_occuped = self.jobs_manager.nb_jobs_in_slots
		self._update_state()
		while self.running_simulation and self.alarm_time != -1:  # and self.jobs_manager.nb_jobs_in_slots == slots_occuped:
			self._checkpoint()
			self._update_state()

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
		output_path = self._output_dir + "/" + str(self.nb_simulation)
		cmd = "batsim -s {} -p {} -w {} -v {} -E -e {}".format(self.protocol_manager.socket_endpoint,
		                                                       self._platform,
		                                                       self._select_workload(),
		                                                       self._verbose,
		                                                       output_path)

		return subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, shell=False)

	def _reset(self):
		self.metrics = {}
		self.time_since_last_new_job = 0
		self.alarm_time = -1
		self._alarm_is_set = False
		self.jobs_manager.reset()
		self.protocol_manager.reset()
		self.resource_manager.reset()

	def _handle_requested_call(self, timestamp):
		self._alarm_is_set = False

	def _handle_job_completed(self, timestamp, data):
		job = self.jobs_manager.on_job_completed(timestamp, data)
		self.resource_manager.release(job)
		self._start_ready_jobs()

	def _handle_job_submitted(self, timestamp, data):
		if data['job']['res'] > self.resource_manager.nb_resources:
			self.protocol_manager.reject_job(data['job_id'])
		else:
			self.time_since_last_new_job = timestamp
			self.jobs_manager.on_job_submitted(timestamp, data['job'])

	def _handle_simulation_ends(self, data):
		self.protocol_manager.acknowledge()
		self.protocol_manager.send_events()
		self.running_simulation = False

		self.metrics['makespan'] = float(data["makespan"])
		self.metrics["mean_waiting_time"] = float(data["mean_waiting_time"])
		self.metrics["mean_turnaround_time"] = float(data["mean_turnaround_time"])
		self.metrics["mean_slowdown"] = float(data["mean_slowdown"])
		self.metrics["max_waiting_time"] = float(data["max_waiting_time"])
		self.metrics["max_turnaround_time"] = float(data["max_turnaround_time"])
		self.metrics["max_slowdown"] = float(data["max_slowdown"])
		self.metrics["energy_consumed"] = float(data["consumed_joules"])
		self.metrics['total_slowdown'] = self.jobs_manager.total_slowdown
		self.metrics['total_turnaround_time'] = self.jobs_manager.total_turnaround_time
		self.metrics['total_waiting_time'] = self.jobs_manager.total_waiting_time
		self._export_metrics()

	def _handle_event(self, event):
		if event.type == "SIMULATION_BEGINS":
			assert not self.running_simulation, "A simulation is already running (is more than one instance of Batsim active?!)"
			self.running_simulation = True
		elif event.type == "SIMULATION_ENDS":
			assert self.running_simulation, "No simulation is currently running"
			self._handle_simulation_ends(event.data)
		elif event.type == "JOB_SUBMITTED":
			self._handle_job_submitted(event.timestamp, event.data)
		elif event.type == "JOB_COMPLETED":
			self._handle_job_completed(event.timestamp, event.data)
		elif event.type == "REQUESTED_CALL":
			self._handle_requested_call(event.timestamp)
		else:
			return

	def _export_metrics(self):
		data = pd.DataFrame(self.metrics, index=[0])
		fn = "{}/{}_{}.csv".format(
			self._output_dir,
			self.nb_simulation,
			"schedule_metrics")
		data.to_csv(fn, index=False)

	def _update_state(self):
		self.protocol_manager.send_events()

		old_time = self.current_time
		events = self.protocol_manager.read_events(
			blocking=not self.running_simulation)

		# New jobs does not need to be updated in this timestep
		# Update jobs if no time has passed does not make sense.
		time_passed = self.current_time - old_time
		if time_passed != 0:
			if self.alarm_time >= self.current_time:
				self.alarm_time = -1

			self.jobs_manager.update_state(time_passed)
			self.resource_manager.update_state(time_passed)
			res = self.resource_manager.shut_down_unused()
			if len(res) != 0:
				self.protocol_manager.set_resource_pstate(res, Resource.PowerState.SHUT_DOWN)

		for event in events:
			self._handle_event(event)

		# Remember to always ack
		if self.running_simulation:
			self.protocol_manager.acknowledge()

	def get_job_slot_state(self):
		s = np.zeros(
			shape=(self.time_window, self.nb_resources * self.job_slots), dtype=np.uint8)

		for i, job in enumerate(self.jobs_manager.job_slots):
			if job != None:
				start_idx = i * self.nb_resources
				end_idx = start_idx + job.requested_resources
				s[0:job.requested_time, start_idx:end_idx] = 1
		return s

	def get_backlog_state(self):
		s = np.zeros(shape=(self.time_window, self.backlog_width), dtype=np.uint8)
		t, i = 0, 0
		nb_jobs = min(self.backlog_width * self.time_window,
		              self.jobs_manager.nb_jobs_in_backlog)
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
		return np.full(shape=self.time_window, fill_value=v, dtype=np.float)

	def _get_image(self):
		shape = (self.time_window, self.nb_resources + self.job_slots * self.nb_resources + self.backlog_width + 1)
		state = np.zeros(shape=shape, dtype=np.float)

		# RESOURCES
		resource_end = self.nb_resources
		state[:, 0:resource_end] = self.resource_manager.get_view()

		# JOB SLOTS
		job_slot_end = self.nb_resources * self.job_slots + self.nb_resources
		state[:, resource_end:job_slot_end] = self.get_job_slot_state()

		# BACKLOG
		backlog_end = self.nb_resources + self.job_slots * self.nb_resources + self.backlog_width
		state[:, job_slot_end:backlog_end] = self.get_backlog_state()

		state[:, -1] = self.get_time_state()

		return state

	def _make_random_dir(self, path):
		num = 1
		while os.path.exists(path + str(num)):
			num += 1

		output_dir = path + str(num)
		os.makedirs(output_dir)
		return output_dir


class GridSimulatorHandler:
	PLATFORM = "platforms/platform_hg_10.xml"
	WORKLOAD_DIR = "workloads"
	OUTPUT_DIR = "results/batsim"

	def __init__(self, job_slots, time_window, backlog_width):
		fullpath = os.path.join(os.path.dirname(__file__), "files")
		if not os.path.exists(fullpath):
			raise IOError("File %s does not exist" % fullpath)

		# self._output_dir = self._make_random_dir(BatsimHandler.OUTPUT_DIR)
		self._platform = os.path.join(fullpath, GridSimulatorHandler.PLATFORM)
		self.time_slice = 1
		workloads_path = os.path.join(
			fullpath, GridSimulatorHandler.WORKLOAD_DIR)
		self._workloads = [workloads_path + "/" +
		                   w for w in os.listdir(workloads_path) if w.endswith('.json')]
		self.nb_simulation = 0
		self.time_window = time_window
		self.job_slots = job_slots
		self.resource_manager = ResourceManager.from_xml(self._platform, self.time_window)
		self.jobs_manager = SchedulerManager(self.nb_resources, self.job_slots)
		self.simulator = GridSimulator(self._workloads, self.jobs_manager)
		self.backlog_width = backlog_width
		self._reset()

	@property
	def nb_jobs_waiting(self):
		return self.jobs_manager.nb_jobs_in_backlog + self.jobs_manager.job_slots.lenght

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
		return self.simulator.current_time

	@property
	def running_simulation(self):
		return self.simulator.running

	def close(self):
		self.simulator.close()

	def start(self):
		assert not self.running_simulation, "A simulation is already running."
		self.simulator.start()
		self._reset()
		self._wait_state_change()
		assert self.running_simulation, "An error ocurred during simulator starting."
		self.nb_simulation += 1

	def schedule(self, job_pos):
		assert self.running_simulation, "Simulation is not running."

		if job_pos != -1:  # Try to schedule job
			job = self.jobs_manager.get_job(job_pos)
			self.resource_manager.allocate(job)
			self.jobs_manager.on_job_allocated(job_pos)
		else:
			self._proceed_time()

		self._start_ready_jobs()

		self._update_state()

	def _wait_state_change(self):
		self._update_state()
		while self.running_simulation and (self.jobs_manager.is_empty or self.resource_manager.is_full()):
			self._proceed_time()
			self._update_state()

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

	def _start_job(self, job):
		self.resource_manager.start_job(job)
		self.jobs_manager.on_job_started(job.id, self.current_time)

	def _reset(self):
		self.metrics = {}
		self.jobs_manager.reset()
		self.resource_manager.reset()

	def _handle_job_completed(self, timestamp, data):
		job = self.jobs_manager.on_job_completed(timestamp, data)
		self.resource_manager.release(job)
		self._start_ready_jobs()

	def _handle_job_submitted(self, timestamp, data):
		if data['job']['res'] <= self.resource_manager.nb_resources:
			self.jobs_manager.on_job_submitted(timestamp, data['job'])
		else:
			self.simulator.reject_job(data['job_id'])

	def _handle_simulation_ends(self, data):
		self.metrics['energy_consumed'] = self.resource_manager.energy_consumption
		self.metrics['makespan'] = self.jobs_manager.last_job.finish_time
		self.metrics['total_slowdown'] = self.jobs_manager.total_slowdown
		self.metrics['mean_slowdown'] = self.jobs_manager.runtime_mean_slowdown
		self.metrics['total_turnaround_time'] = self.jobs_manager.total_turnaround_time
		self.metrics[
			'mean_turnaround_time'] = self.jobs_manager.total_turnaround_time / self.jobs_manager.nb_jobs_submitted
		self.metrics['total_waiting_time'] = self.jobs_manager.total_waiting_time
		self.metrics['mean_waiting_time'] = self.jobs_manager.total_waiting_time / self.jobs_manager.nb_jobs_submitted

	# self._export_metrics()

	def _handle_event(self, event):
		if event.type == "SIMULATION_ENDS":
			self._handle_simulation_ends(event.data)
		elif event.type == "JOB_SUBMITTED":
			self._handle_job_submitted(event.timestamp, event.data)
		elif event.type == "JOB_COMPLETED":
			self._handle_job_completed(event.timestamp, event.data)
		else:
			raise Exception("Unknown event type {}".format(event.type))

	def _get_resources_from_json(self, data):
		resources = []
		for alloc in data.split(" "):
			nodes = alloc.split("-")
			if len(nodes) == 2:
				resources.extend(range(int(nodes[0]), int(nodes[1]) + 1))
			else:
				resources.append(int(nodes[0]))
		return resources

	def _proceed_time(self):
		self.simulator.proceed_time(1)
		self.jobs_manager.update_state(1)
		self.resource_manager.update_state(1)
		self.resource_manager.shut_down_unused()

	def _update_state(self):
		events = self.simulator.read_events()
		for event in events:
			self._handle_event(event)

	def get_job_slot_state(self):
		s = np.zeros(shape=(self.time_window, self.nb_resources * self.job_slots), dtype=np.uint8)

		for i, job in enumerate(self.jobs_manager.job_slots):
			if job is not None:
				start_idx = i * self.nb_resources
				end_idx = start_idx + job.requested_resources
				s[0:job.requested_time, start_idx:end_idx] = 1
		return s

	#def get_job_slot_state(self):
	#	state = np.zeros(shape=(self.time_window, self.nb_resources * self.job_slots), dtype=np.float)
#
	#	for i, job in enumerate(self.jobs_manager.job_slots):
	#		if job is not None:
	#			start_idx = i * self.nb_resources
	#			end_idx = start_idx + job.requested_resources
	#			frac_time, req_time = math.modf(job.requested_time / self.time_slice)
	#			state[0:int(req_time), start_idx:end_idx] = 1.
	#			if frac_time != 0:
	#				state[int(req_time), start_idx:end_idx] = frac_time
	#	return state

	def get_backlog_state(self):
		s = np.zeros(shape=(self.time_window, self.backlog_width), dtype=np.uint8)
		t, i = 0, 0
		nb_jobs = min(self.backlog_width * self.time_window, self.jobs_manager.nb_jobs_in_backlog)
		for _ in range(nb_jobs):
			s[t, i] = 1
			i += 1
			if i == self.backlog_width:
				i = 0
				t += 1
		return s

	def get_time_state(self):
		v = self.simulator.time_since_last_new_job / float(self.simulator.max_tracking_time_since_last_job)
		return np.full(shape=self.time_window, fill_value=v, dtype=np.float)

	def get_state(self):
		return self._get_image()

	def get_resource_state(self):
		return self.resource_manager.get_view()
		#resource_state = self.resource_manager.get_view()
		#state = np.zeros(shape=(self.time_window, self.nb_resources))
		#start = 0
		#for init in range(0, resource_state.shape[0], self.time_slice):
		#	state[start, 0:self.nb_resources] = np.sum(resource_state[init:init + self.time_slice, 0:self.nb_resources],
		#	                                           axis=0) / self.time_slice
		#	start += 1
#
		#return state

	def _get_image(self):
		assert self.time_window % self.time_slice == 0
		shape = (self.time_window, self.nb_resources + self.job_slots * self.nb_resources + self.backlog_width + 1)
		state = np.zeros(shape=shape, dtype=np.float)

		# RESOURCES
		resource_end = self.nb_resources
		state[:, 0:resource_end] = self.get_resource_state()

		# JOB SLOTS
		job_slot_end = self.nb_resources * self.job_slots + resource_end

		#for i, job in enumerate(self.jobs_manager.job_slots):
		#	if job is not None:
		#		state[i, 0] = job.requested_resources / self.nb_resources
		#		state[i, 1] = job.requested_time / self.time_window

		state[:, resource_end:job_slot_end] = self.get_job_slot_state()

		# BACKLOG
		backlog_end = job_slot_end + self.backlog_width
		state[:, job_slot_end:backlog_end] = self.get_backlog_state()

		state[:, -1] = self.get_time_state()

		#state = np.expand_dims(state, axis=2)

		return state
