import numpy as np
from xml.dom import minidom
from enum import Enum


class Resource:
	class State(Enum):
		SLEEPING = 'sleeping'
		IDLE = 'idle'
		COMPUTING = 'computing'

	class PowerState(Enum):
		SHUT_DOWN = 0
		NORMAL = 1

	def __init__(self, id, state, pstate, name, profiles):
		assert isinstance(state, Resource.State)
		assert isinstance(pstate, Resource.PowerState)
		assert isinstance(profiles, dict)
		assert isinstance(id, int)
		self._queue = list()
		self.state = state
		self.pstate = pstate
		self.name = name
		self.id = id
		self.profiles = profiles
		self.energy_to_turn_on = 0
		self.max_watt = self.profiles[Resource.PowerState.NORMAL]['watt_comp']
		self.min_watt = self.profiles[Resource.PowerState.SHUT_DOWN]['watt_idle']
		self.speed = self.profiles[Resource.PowerState.NORMAL]['speed']

	@staticmethod
	def load(id, data):
		name = data.getAttribute('id')
		host_speed = data.getAttribute('speed').split(',')
		host_power_state = Resource.PowerState(int(data.getAttribute('pstate')))
		assert not host_power_state == Resource.PowerState.SHUT_DOWN, "Not supported yet."
		host_state = Resource.State.SLEEPING if host_power_state == Resource.PowerState.SHUT_DOWN else Resource.State.IDLE
		host_watts = data.getElementsByTagName('prop')[0].getAttribute('value').split(',')
		assert "Mf" in host_speed[-1], "Speed should be in Mega Flops (Mf)"

		profiles = {}
		for i, speed in enumerate(host_speed):
			(idle, comp) = host_watts[i].split(":")
			if i == 0:
				assert idle == comp, "Idle and Comp. power states should be equal for resource state shut down (0)"

			profiles[Resource.PowerState(i)] = {
				'speed': float(speed.replace("Mf", "")),
				'watt_idle': float(idle),
				'watt_comp': float(comp)
			}
		return Resource(id, host_state, Resource.PowerState(host_power_state), name, profiles)

	@property
	def is_sleeping(self):
		return self.state == Resource.State.SLEEPING

	@property
	def is_computing(self):
		return self.state == Resource.State.COMPUTING

	@property
	def current_watt(self):
		curr_prop = self.profiles[self.pstate]
		return curr_prop['watt_comp'] if self.is_computing else curr_prop['watt_idle']

	@property
	def queue(self):
		return sorted(self._queue, key=lambda j: j.time_left_to_start) if self._queue else list()

	def get_reserved_time(self):
		return self._queue[0].remaining_time if self._queue else 0

	def get_energy_consumption(self, time_passed):
		energy = (self.current_watt * time_passed) + self.energy_to_turn_on
		self.energy_to_turn_on = 0
		return energy

	def get_job(self):
		return min(self._queue, key=(lambda j: j.time_left_to_start)) if self._queue else None

	def sleep(self):
		assert not self.is_computing, "Cannot sleep resource while computing!"
		self.state = Resource.State.SLEEPING
		self.pstate = Resource.PowerState.SHUT_DOWN

	def wake_up(self):
		if self.is_sleeping:
			self.energy_to_turn_on += self.min_watt * 2
		self.state = Resource.State.IDLE
		self.pstate = Resource.PowerState.NORMAL

	def start_computing(self):
		assert self.state == Resource.State.IDLE
		self.state = Resource.State.COMPUTING

	def get_view(self):
		if not self._queue:
			return np.array(list())
		last_job = max(self._queue, key=(lambda j: j.time_left_to_start))
		state = np.zeros(shape=int(last_job.time_left_to_start) + int(last_job.remaining_time), dtype=np.int8)
		for j in self._queue:
			state[j.time_left_to_start:j.time_left_to_start + int(j.remaining_time)] = 1
		return state

	def reserve(self, job):
		self._queue.append(job)

	def release(self, job):
		self.state = Resource.State.IDLE
		self._queue.remove(job)

	def reset(self):
		self.energy_to_turn_on = 0
		self.state = Resource.State.IDLE
		self.pstate = Resource.PowerState.NORMAL
		self._queue.clear()


class ResourceManager:
	def __init__(self, resources):
		self.nb_resources = len(resources)
		self._resources = sorted(resources, key=lambda r: r.id)
		self.energy_consumed = 0
		self.colormap = np.arange(1 / 40., 1, 1 / 40.).tolist()
		np.random.shuffle(self.colormap)

	@property
	def resources(self):
		return self._resources

	# @property
	# def max_resource_speed(self):
	#	return max(self._resources.items(), key=(lambda item: item[1].max_speed))[1].max_speed

	@staticmethod
	def load(platform_fn):
		platform = minidom.parse(platform_fn)
		hosts = platform.getElementsByTagName('host')
		hosts.sort(key=lambda x: x.attributes['id'].value)
		resources = list()
		id = 0
		for host in hosts:
			if host.getAttribute('id') != 'master_host':
				resources.append(Resource.load(id, host))
				id += 1

		return ResourceManager(resources)

	def is_full(self):
		return not any(not r.is_computing for r in self._resources)

	def is_available(self, res_idxs):
		return not any(self._resources[i].is_computing for i in res_idxs)

	def status(self):
		computing, idle, sleeping = 0, 0, 0
		for r in self._resources:
			if r.is_sleeping:
				sleeping += 1
			elif r.is_computing:
				computing += 1
			else:
				idle += 1
		return computing, idle, sleeping

	def get_view(self):
		return np.array([r.get_view() for r in self._resources])

	def reset(self):
		np.random.shuffle(self.colormap)
		self.energy_consumed = 0
		for r in self._resources: r.reset()

	def shut_down_unused(self):
		res = list()
		for res in self._resources:
			if not res.is_computing and not res.is_sleeping:
				res.append(res.id)
				res.sleep()
		return res

	def _reserve_resources(self, job):
		resources, count, min_speed = list(), 0, np.inf
		for i, r in enumerate(self._resources):
			if not r.is_computing:
				resources.append(i)
				count += 1

			if count == job.requested_resources:
				for j in resources:
					self._resources[j].reserve(job)
					if self._resources[j].speed < min_speed:
						min_speed = self._resources[j].speed
				return resources, min_speed

		raise UnavailableResourcesError("There is no resource available for this job.")

	def update_state(self, time_passed):
		assert time_passed != 0
		for r in self._resources:
			self.energy_consumed += r.get_energy_consumption(time_passed)

	def get_jobs(self):
		jobs = dict()
		for r in self._resources:
			j = r.get_job()
			if j is not None and j.id not in jobs:
				jobs[j.id] = j
		return jobs.values()

	def _select_color(self):
		c = self.colormap.pop(0)
		self.colormap.append(c)
		return c

	def allocate_job_and_throw(self, job):
		res_idx, slowest_speed = self._reserve_resources(job)
		job.allocation = res_idx
		job.time_left_to_start = 0
		job.expected_exec_time = job.cpu / 1000000.0 / float(slowest_speed)  # Convert to Mega Flops

	def start_job(self, job):
		res = list()
		for r in job.allocation:
			resource = self._resources[r]
			if resource.is_sleeping:
				res.append(r)
			resource.start_computing()
		return res

	def release(self, job):
		for r in job.allocation:
			self._resources[r].release(job)

	def set_sleep(self, res_idx):
		for i in res_idx:
			self._resources[i].sleep()

	def wake_up(self, res_idx):
		res = list()
		for i in res_idx:
			if self._resources[i].is_sleeping:
				self._resources[i].wake_up()
				res.append(i)
		return res


class UnavailableResourcesError(Exception):
	pass
