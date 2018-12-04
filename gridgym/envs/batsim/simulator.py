import json
import numpy as np
import os
import time as tm
import pandas as pd
import subprocess
import math
from sortedcontainers import SortedList
from .job import Job, JobManager
from .network import BatsimEvent, BatsimProtocolHandler
from .resource import ResourceManager


class GridSimulator:
	def __init__(self, jobs_manager):
		self.jobs_manager = jobs_manager
		self.workload = []
		self.workload_nb_jobs = -1
		self.jobs_submmited = -1
		self.jobs_completed = -1
		self.running = False
		self.current_time = -1

	def close(self):
		self.workload = []
		self.workload_nb_jobs = -1
		self.jobs_submmited = -1
		self.jobs_completed = -1
		self.running = False
		self.current_time = -1

	def get_jobs_completed(self, time):
		for job in self.jobs_manager.jobs_running:
			if job.remaining_time == 0 or job.runtime >= job.expected_exec_time:
				yield job

	def get_jobs_submmited(self, time):
		while len(self.workload) > 0 and self.workload[0].submit_time == time:
			yield self.workload.pop(0)

	def reject_job(self, job_id):
		self.jobs_completed += 1

	def get_job_submitted_event(self, time, job):
		data = dict(
			job_id=job.id,
			job=dict(
				profile=job.profile,
				res=job.requested_resources,
				id=job.id,
				subtime=job.submit_time,
				walltime=job.requested_time))
		return BatsimEvent(time, "JOB_SUBMITTED", data)

	def get_job_completed_event(self, time, job):
		if job.expected_exec_time > job.runtime:
			job_state = Job.State.KILLED
			kill_reason = "WALLTIME REACHED"
		else:
			job_state = Job.State.COMPLETED
			kill_reason = ""

		data = dict(
			job_id=job.id,
			job_state=job_state,
			return_code=0,
			kill_reason=kill_reason,
			alloc=job.allocation)
		return BatsimEvent(time, "JOB_COMPLETED", data)

	@property
	def simulation_ended(self):
		return self.jobs_submmited == self.workload_nb_jobs and self.jobs_completed == self.workload_nb_jobs

	def proceed_time(self, t):
		self.current_time += t

	def start(self, workload):
		def get_jobs():
			with open(workload, 'r') as f:
				data = json.load(f)
				jobs = SortedList(key=lambda t: t.submit_time)
				for j in data['jobs']:
					jobs.add(Job(
						j['id'],
						j['subtime'],
						j['walltime'],
						j['res'],
						j['profile'],
						-1,
						"UNKNOWN"))
			return jobs

		self.workload = get_jobs()
		self.workload_nb_jobs = len(self.workload)
		self.current_time = self.workload[0].submit_time
		self.jobs_submmited = 0
		self.jobs_completed = 0
		self.running = True

	def read_events(self):
		assert self.running
		events = []

		for j in self.get_jobs_submmited(self.current_time):
			self.jobs_submmited += 1
			events.append(self.get_job_submitted_event(self.current_time, j))

		for j in self.get_jobs_completed(self.current_time):
			self.jobs_completed += 1
			events.append(self.get_job_completed_event(self.current_time, j))

		if self.simulation_ended:
			self.running = False
			events.append(BatsimEvent(self.current_time, "SIMULATION_ENDS", dict()))

		return events


class SimulatorHandler:
	PLATFORM_FN = "platforms/platform_hg_10.xml"
	WORKLOAD_DIR = "workloads"
	OUTPUT_DIR = "results"

	def __init__(self, nb_job_slots, time_slice, backlog_width):
		self.files_path = os.path.join(os.path.dirname(__file__), "files")
		if not os.path.exists(self.files_path):
			raise IOError("File %s does not exist" % self.files_path)

		os.makedirs(SimulatorHandler.OUTPUT_DIR, exist_ok=True)
		self.time_slice = time_slice
		self.nb_job_slots = nb_job_slots
		self.backlog_width = backlog_width
		self.img_time_window = 60
		self.nb_simulation = 0
		self.max_tracking_time_since_last_job = 10
		self._workload_idx = 0
		self.time_since_last_new_job = 0
		self.metrics = {}
		self._workloads = self._get_workloads(self.files_path)
		self._platform = os.path.join(self.files_path, SimulatorHandler.PLATFORM_FN)
		self.resource_manager = ResourceManager.load(self._platform)
		self.job_manager = JobManager(self.nb_job_slots)

	@property
	def nb_jobs_waiting(self):
		return self.job_manager.nb_jobs_in_queue + self.job_manager.nb_jobs_in_slots

	@property
	def nb_jobs_running(self):
		return self.job_manager.nb_jobs_running

	@property
	def nb_jobs_submitted(self):
		return self.job_manager.nb_jobs_submitted

	@property
	def nb_jobs_completed(self):
		return self.job_manager.nb_jobs_finished

	@property
	def nb_resources(self):
		return self.resource_manager.nb_resources

	@property
	def current_time(self):
		raise NotImplementedError()

	@property
	def is_running(self):
		raise NotImplementedError()

	def _get_workloads(self, files_path):
		workloads_path = os.path.join(files_path, SimulatorHandler.WORKLOAD_DIR)
		return [workloads_path + "/" + w for w in os.listdir(workloads_path) if w.endswith('.json')]

	def schedule(self, job_pos):
		assert self.is_running, "Simulation is not running."
		if job_pos != -1:
			job = self.job_manager.get_job_and_throw(job_pos)
			self.resource_manager.allocate_job_and_throw(job)
			self.job_manager.on_job_allocated(job_pos)
			self._start_ready_jobs()
			return

		self._proceed_time()

		self._start_ready_jobs()

		self._wait_state_change()

	def _wait_state_change(self):
		raise NotImplementedError()

	def _proceed_time(self):
		raise NotImplementedError()

	def _start_ready_jobs(self):
		for job in self.resource_manager.get_jobs():
			if job.state != Job.State.RUNNING and job.time_left_to_start == 0 and self.resource_manager.is_available(
					job.allocation):
				self._start_job(job)

	def _handle_job_completed(self, timestamp, data):
		job = self.job_manager.on_job_completed(data['job_id'], timestamp)
		self.resource_manager.release(job)
		self._start_ready_jobs()

	def _handle_job_submitted(self, timestamp, data):
		if data['job']['res'] <= self.resource_manager.nb_resources:
			cpu, _, tpe = self.job_manager.get_profile(data['job']["profile"])
			self.job_manager.on_job_submitted(
				Job(
					data['job']["id"],
					data['job']["subtime"],
					data['job'].get("walltime", -1),
					data['job']["res"],
					data['job']["profile"],
					cpu,
					tpe),
				timestamp)
			self.time_since_last_new_job = timestamp
		else:
			self._reject_job(data['job_id'])

	def _handle_simulation_ends(self, data):
		self.metrics = dict(
			makespan=self.job_manager.last_job.finish_time,
			mean_waiting_time=self.job_manager.total_waiting_time / self.job_manager.nb_jobs_submitted,
			mean_turnaround_time=self.job_manager.total_turnaround_time / self.job_manager.nb_jobs_submitted,
			mean_slowdown=self.job_manager.runtime_mean_slowdown,
			max_waiting_time=self.job_manager.max_waiting_time,
			max_turnaround_time=self.job_manager.max_turnaround_time,
			max_slowdown=self.job_manager.max_slowdown_time,
			energy_consumed=self.resource_manager.energy_consumed,
			nb_jobs_killed=self.job_manager.nb_jobs_killed,
			nb_jobs_finished=self.job_manager.nb_jobs_finished,
			nb_jobs_success=self.job_manager.nb_jobs_success,
			total_slowdown=self.job_manager.total_slowdown,
			total_turnaround_time=self.job_manager.total_turnaround_time,
			total_waiting_time=self.job_manager.total_waiting_time
		)
		self._export_metrics()

	def _handle_simulation_begins(self, data):
		pass

	def _handle_requested_call(self, timestamp):
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

	def get_job_state_img(self):
		job_state = np.zeros(shape=(self.img_time_window, self.nb_resources * self.nb_job_slots), dtype=np.float)

		for i, job in enumerate(self.job_manager.job_slots):
			if job is not None:
				fraction, whole = math.modf(job.requested_time / float(self.time_slice))
				job_state[i * self.nb_resources:min(int(whole), self.img_time_window), 0:job.requested_resources] = 1.
				if fraction != 0 and int(whole) < self.img_time_window:
					job_state[int(whole), 0:job.requested_resources] = fraction
		return job_state

	def get_backlog_state_img(self):
		state = np.zeros(shape=(self.img_time_window, self.backlog_width), dtype=np.uint8)
		t, i = 0, 0
		nb_jobs = min(self.backlog_width * self.img_time_window, self.job_manager.nb_jobs_in_queue)
		for _ in range(nb_jobs):
			state[t, i] = 1
			i += 1
			if i == self.backlog_width:
				i = 0
				t += 1
		return state

	def get_time_state_img(self):
		diff = min(self.max_tracking_time_since_last_job, self.current_time - self.time_since_last_new_job)
		v = diff / float(self.max_tracking_time_since_last_job)
		return np.full(shape=self.img_time_window, fill_value=v, dtype=np.float)

	def get_resource_state_img(self):
		res_state = np.zeros(shape=(self.img_time_window, self.nb_resources), dtype=np.float)
		for i, r in enumerate(self.resource_manager.get_view()):
			time = 0
			for t in range(0, min(self.img_time_window * self.time_slice + 1, len(r)), self.time_slice):
				res_state[time, i] = np.sum(r[t:t + self.time_slice]) / self.time_slice
				time += 1

		return res_state

	def get_state(self):
		state = np.zeros(shape=(self.nb_resources + self.nb_job_slots * 2), dtype=np.float)
		state[0:self.nb_resources] = [res.get_reserved_time() for res in self.resource_manager.resources]

		i = self.nb_resources
		for j in self.job_manager.job_slots:
			if j is not None:
				state[i] = j.requested_resources
				state[i + 1] = j.requested_time
			i += 2
		return state

	def select_workload(self):
		if len(self._workloads) == self._workload_idx:
			self._workload_idx = 0
			np.random.shuffle(self._workloads)

		workload = self._workloads[self._workload_idx]
		self._workload_idx += 1
		return workload

	def start(self, workload_fn):
		self.job_manager.load(workload_fn)

	def close(self):
		raise NotImplementedError()

	def _start_job(self, job):
		self.resource_manager.start_job(job)
		self.job_manager.on_job_started(job.id, self.current_time)

	def _reject_job(self, job_id):
		raise NotImplementedError()

	def reset(self):
		self.metrics = {}
		self.time_since_last_new_job = 0
		self.job_manager.reset()
		self.resource_manager.reset()

	def _export_metrics(self):
		fn = "{}/env_{}_metrics.csv".format(SimulatorHandler.OUTPUT_DIR, self.nb_simulation)
		pd.DataFrame(self.metrics, index=[0]).to_csv(fn, index=False)


class BatsimHandler(SimulatorHandler):
	USE_DOCKER = False

	def __init__(self, nb_job_slots, time_slice, backlog_width, verbose='quiet'):
		self.protocol_manager = BatsimProtocolHandler()
		self.running_simulation = False
		self._alarm_is_set = False
		self._simulator_process = None
		self.verbose = verbose
		self._workload_idx = 0
		super(BatsimHandler, self).__init__(nb_job_slots, time_slice, backlog_width)

	@property
	def current_time(self):
		return self.protocol_manager.current_time

	@property
	def is_running(self):
		return self.running_simulation

	def close(self):
		self.protocol_manager.close()
		self.running_simulation = False
		if self._simulator_process is not None:
			self._simulator_process.terminate()
			self._simulator_process.wait()
			self._simulator_process = None

	def start(self, workload_fn=None):
		assert not self.is_running, "A simulation is already running."
		self.reset()
		self._simulator_process, workload_fn = self._start_simulator(workload_fn)
		super(BatsimHandler, self).start(workload_fn)
		self.protocol_manager.start()
		while self.job_manager.is_empty:
			self._handle_events()

		assert self.is_running, "An error ocurred during simulator starting."
		self.nb_simulation += 1

	def _proceed_time(self):
		if not self._alarm_is_set:
			self.protocol_manager.set_alarm(self.current_time + 1.0001)
			self._alarm_is_set = True

	def _wait_state_change(self):
		self._handle_events()
		while self.is_running and (self._alarm_is_set or self.job_manager.is_empty):
			self._proceed_time()
			self._handle_events()

	def _start_job(self, job):
		super(BatsimHandler, self)._start_job(job)
		self.protocol_manager.start_job(job.id, job.allocation)

	def _start_simulator(self, workload_fn):
		platform_file = self._platform
		workload_file = self.select_workload() if workload_fn is None else workload_fn

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
		return subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, shell=False), workload_file

	def reset(self):
		super(BatsimHandler, self).reset()
		self._alarm_is_set = False
		self.protocol_manager.reset()

	def _reject_job(self, job_id):
		self.protocol_manager.reject_job(job_id)

	def _handle_events(self):
		self.protocol_manager.send_events()

		old_time = self.current_time
		events = self.protocol_manager.read_events(blocking=not self.running_simulation)

		# New jobs does not need to be updated in this timestep
		# Update jobs if no time has passed does not make sense.
		time_passed = self.current_time - old_time
		if time_passed != 0:
			self.job_manager.update_state(time_passed)
			self.resource_manager.update_state(time_passed)

		for event in events:
			self._handle_event(event)

		# Remember to always ack
		if self.running_simulation:
			self.protocol_manager.acknowledge()

	def _handle_requested_call(self, timestamp):
		self._alarm_is_set = False

	def _handle_simulation_ends(self, data):
		assert self.is_running, "No simulation is currently running"
		self.protocol_manager.acknowledge()
		self.protocol_manager.send_events()
		self.running_simulation = False
		self.metrics = dict(
			makespan=float(data["makespan"]),
			mean_waiting_time=float(data["mean_waiting_time"]),
			mean_turnaround_time=float(data["mean_turnaround_time"]),
			mean_slowdown=float(data["mean_slowdown"]),
			max_waiting_time=float(data["max_waiting_time"]),
			max_turnaround_time=float(data["max_turnaround_time"]),
			max_slowdown=float(data["max_slowdown"]),
			energy_consumed=float(data["consumed_joules"]),
			nb_jobs_killed=data["nb_jobs_killed"],
			nb_jobs_finished=data["nb_jobs_finished"],
			nb_jobs_success=data["nb_jobs_success"],
			total_slowdown=self.job_manager.total_slowdown,
			total_turnaround_time=self.job_manager.total_turnaround_time,
			total_waiting_time=self.job_manager.total_waiting_time
		)
		tm.sleep(2)
		self._export_metrics()

	def _handle_simulation_begins(self, data):
		assert not self.is_running, "A simulation is already running (is more than one instance of Batsim active?!)"
		self.running_simulation = True



class GridSimulatorHandler(SimulatorHandler):
	def __init__(self, nb_job_slots, time_slice, backlog_width):
		super(GridSimulatorHandler, self).__init__(nb_job_slots, time_slice, backlog_width)
		self.simulation_manager = GridSimulator(self.job_manager)

	@property
	def current_time(self):
		return self.simulation_manager.current_time

	@property
	def is_running(self):
		return self.simulation_manager.running

	def start(self, workload_fn=None):
		assert not self.is_running, "A simulation is already running."
		workload = workload_fn if workload_fn is not None else self.select_workload()
		self.simulation_manager.start(workload)
		super(GridSimulatorHandler, self).start(workload)
		self.reset()
		self._wait_state_change()
		assert self.is_running, "An error ocurred during simulator starting."
		self.nb_simulation += 1

	def close(self):
		self.simulation_manager.close()

	def _reject_job(self, job_id):
		self.simulation_manager.reject_job(job_id)

	def _wait_state_change(self):
		self._handle_events()
		while self.is_running and self.job_manager.is_empty:
			self._proceed_time()
			self._handle_events()

	def _proceed_time(self):
		self.simulation_manager.proceed_time(1)
		self.job_manager.update_state(1)
		self.resource_manager.update_state(1)

	def _handle_events(self):
		events = self.simulation_manager.read_events()
		for event in events:
			self._handle_event(event)
