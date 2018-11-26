import math
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

	def __init__(self, id, state, pstate, name, profiles, time_window):
		assert isinstance(state, Resource.State)
		assert isinstance(pstate, Resource.PowerState)
		assert isinstance(profiles, dict)
		assert isinstance(id, int)
		self.queue = []
		self.time_window = time_window
		self.state = state
		self.pstate = pstate
		self.name = name
		self.id = id
		self.profiles = profiles
		self.energy_to_turn_on = 0
		self.max_watt = self.profiles[Resource.PowerState.NORMAL]['watt_comp']
		self.min_watt = self.profiles[Resource.PowerState.SHUT_DOWN]['watt_idle']
		self.max_speed = self.profiles[Resource.PowerState.NORMAL]['speed']

	@staticmethod
	def from_xml(id, data, time_window):
		name = data.getAttribute('id')
		host_speed = data.getAttribute('speed').split(',')
		host_power_state = Resource.PowerState(int(data.getAttribute('pstate')))
		host_state = Resource.State.SLEEPING if host_power_state == Resource.PowerState.SHUT_DOWN else Resource.State.IDLE
		host_watts = data.getElementsByTagName(
			'prop')[0].getAttribute('value').split(',')
		assert "Mf" in host_speed[-1], "Speed is not in Mega Flops"

		profiles = {}
		for i, speed in enumerate(host_speed):
			(idle, comp) = host_watts[i].split(":")
			if i == 0:
				assert idle == comp, "Idle and Comp power states should be equal for resource state 0"

			profiles[Resource.PowerState(i)] = {
				'speed': float(speed.replace("Mf", "")),
				'watt_idle': float(idle),
				'watt_comp': float(comp)
			}
		return Resource(id, host_state, Resource.PowerState(host_power_state), name, profiles, time_window)

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

	def get_reserved_time(self):
		return 0 if len(self.queue) == 0 else self.queue[0].remaining_time

	def get_energy_consumption(self, time_passed):
		energy = (self.current_watt * time_passed) + self.energy_to_turn_on
		self.energy_to_turn_on = 0
		return energy

	def get_job(self):
		return min(self.queue, key=(lambda j: j.time_left_to_start)) if self.queue else None

	def get_queue(self):
		return sorted(self.queue, key=lambda j: j.time_left_to_start) if self.queue else list()

	def sleep(self):
		# assert not self.is_computing, "Cannot sleep resource while computing!"
		self.state = Resource.State.SLEEPING
		self.pstate = Resource.PowerState.SHUT_DOWN

	def wake_up(self):
		if self.is_sleeping:
			self.energy_to_turn_on += self.min_watt * 2
		self.state = Resource.State.IDLE
		self.pstate = Resource.PowerState.NORMAL

	def start_computing(self):
		self.wake_up()
		self.pstate = Resource.PowerState.NORMAL
		self.state = Resource.State.COMPUTING

	def get_view(self):
		# if self.is_sleeping:  # SLEEP
		view = np.zeros(shape=self.time_window, dtype=np.float)
		# else:
		#    view = np.full(shape=self.time_window,
		#                   fill_value=127, dtype=np.uint8)

		if len(self.queue) == 0:
			return view

		# view[self.queue[0].time_left_to_start:self.time_window] = 127

		for j in self.queue:
			end = j.time_left_to_start + int(j.remaining_time)
			view[j.time_left_to_start:end] = 1.0

		return view

	def reserve(self, job):
		self.queue.append(job)

	def release(self, job):
		self.state = Resource.State.IDLE
		self.queue.remove(job)

	def reset(self):
		self.energy_to_turn_on = 0
		self.sleep()
		self.queue.clear()


class ResourceManager:
	def __init__(self, resources, time_window):
		assert isinstance(resources, dict)
		self.nb_resources = len(resources)
		self.time_window = time_window
		self.resources = resources
		self.energy_consumption = 0
		self.max_energy_usage = 0

		self.colormap = np.arange(1 / 40., 1, 1 / 40.).tolist()
		np.random.shuffle(self.colormap)

	@property
	def max_resource_speed(self):
		return max(self.resources.items(), key=(lambda item: item[1].max_speed))[1].max_speed

	@staticmethod
	def from_xml(platform_fn, time_window):
		platform = minidom.parse(platform_fn)
		hosts = platform.getElementsByTagName('host')
		hosts.sort(key=lambda x: x.attributes['id'].value)
		resources = {}
		id = 0
		for host in hosts:
			if host.getAttribute('id') != 'master_host':
				resources[id] = Resource.from_xml(
					id, host, time_window=time_window)
				id += 1

		return ResourceManager(resources, time_window)

	def is_full(self):
		return not any(not r.is_computing for k, r in self.resources.items())

	def is_available(self, resources):
		return not any(self.resources[r].is_computing for r in resources)

	def nb_resources_state(self):
		computing, idle, sleeping = 0, 0, 0
		for _, r in self.resources.items():
			if r.is_sleeping:
				sleeping += 1
			elif r.is_computing:
				computing += 1
			else:
				idle += 1
		return computing, idle, sleeping

	def get_view(self):
		state = np.zeros(shape=(self.time_window, self.nb_resources), dtype=np.float)
		for k, res in self.resources.items():
			state[:, k] = res.get_view()

		return state

	def reset(self):
		np.random.shuffle(self.colormap)
		self.energy_consumption = 0
		self.max_energy_usage = 0
		for _, r in self.resources.items():
			r.reset()

	def shut_down_unused(self):
		res = []
		for i, r in self.resources.items():
			if len(r.queue) == 0 and not r.is_sleeping:
				res.append(i)
				r.sleep()

		return res

	def _select_resources(self, nb_res, time):
		state = self.get_view()
		for t in range(self.time_window-time+1):
			avail_resources = [r for r in range(self.nb_resources) if np.count_nonzero(state[t:t+time, r]) == 0]
			if len(avail_resources) >= nb_res:
				return avail_resources[0:nb_res], t

		raise UnavailableResourcesError("There is no resource available.")

	# def _select_resources(self, nb_res, time):
	#    avail_resources = [k for k, r in self.resources.items() if not r.is_computing]
	#    if len(avail_resources) >= nb_res:
	#        return avail_resources[0:nb_res], 0
	#
	#    raise UnavailableResourcesError("There is no resource available.")

	def update_state(self, time_passed):
		assert time_passed != 0
		self.max_energy_usage = 0
		for _, r in self.resources.items():
			self.energy_consumption += r.get_energy_consumption(time_passed)
			self.max_energy_usage += r.max_watt * time_passed

	def get_jobs(self):
		jobs = dict()
		for _, r in self.resources.items():
			j = r.get_job()
			if j is not None and j.id not in jobs:
				jobs[j.id] = j

		return jobs.values()

	def _select_color(self):
		c = self.colormap.pop(0)
		self.colormap.append(c)
		return c

	def allocate(self, job):
		res, time = self._select_resources(
			job.requested_resources, job.requested_time)
		job.allocation = res
		job.time_left_to_start = time

		if job.color is None:
			job.color = self._select_color()

		for r in res:
			self.resources[r].reserve(job)

	def start_job(self, job):
		res = []
		for r in job.allocation:
			resource = self.resources[r]
			if resource.is_sleeping:
				res.append(r)
			resource.start_computing()
		return res

	def release(self, job):
		for r in job.allocation:
			self.resources[r].release(job)

	def get_resources(self):
		resources = np.empty((self.nb_resources,), dtype=object)
		for k, value in self.resources.items():
			resources[k] = value

		return resources

	def set_sleep(self, res_ids):
		for i in res_ids:
			self.resources[i].sleep()

	def wake_up(self, res_ids):
		res = []
		for i in res_ids:
			if self.resources[i].is_sleeping:
				self.resources[i].wake_up()
				res.append(i)
		return res


class InvalidPowerStateError(Exception):
	pass


class UnavailableResourcesError(Exception):
	pass
