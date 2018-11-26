import gym
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.pyplot as plt
from .batsim import GridSimulatorHandler, UnavailableResourcesError, InvalidJobError
import numpy as np


class GridEnv(gym.Env):
	JOB_SLOTS = 10
	TIME_WINDOW = 20
	BACKLOG_WIDTH = 3

	def __init__(self):
		self.simulator = GridSimulatorHandler(GridEnv.JOB_SLOTS, GridEnv.TIME_WINDOW, GridEnv.BACKLOG_WIDTH)
		self.action_space = spaces.Discrete(GridEnv.JOB_SLOTS + 1)
		state = self._get_obs()
		self.observation_space = spaces.Box(low=0, high=1, shape=state.shape, dtype=state.dtype)

	def step(self, action):
		assert self.simulator.is_running, "Simulation is not running."
		mean_slow_before = self.simulator.jobs_manager.runtime_mean_slowdown

		try:
			self.simulator.schedule(action - 1)
		except (UnavailableResourcesError, InvalidJobError):
			self.simulator.schedule(-1)

		obs = self._get_obs()
		reward = -1 * (self.simulator.jobs_manager.runtime_mean_slowdown - mean_slow_before)
		done = not self.simulator.is_running
		info = self._get_info()

		return obs, reward, done, info

	def reset(self):
		self.simulator.close()
		self.simulator.start()
		return self._get_obs()

	def render(self, mode='human'):
		if mode == 'human':
			self._plot()
		elif mode == 'console':
			self._print()
		else:
			self._plot()

	def close(self):
		self.simulator.close()

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _get_info(self):
		return dict() if self.simulator.is_running else self.simulator.metrics

	def _get_obs(self):
		return self.simulator.get_state()

	def _print(self):
		stats = "\rSubmitted: {:5} Completed: {:5} | Running: {:5} Waiting: {:5}".format(
			self.simulator.nb_jobs_submitted,
			self.simulator.nb_jobs_completed,
			self.simulator.nb_jobs_running,
			self.simulator.nb_jobs_waiting)
		print(stats, end="", flush=True)

	def _plot(self):
		def plot_resource_state():
			resource_state = self.simulator.get_resource_state()
			plt.subplot(1, 1 + GridEnv.JOB_SLOTS + 2, 1)
			plt.imshow(resource_state, interpolation='nearest', vmin=0, vmax=1, aspect='auto')
			ax = plt.gca()
			ax.set_xticks(range(self.simulator.nb_resources))
			ax.set_yticks(range(GridEnv.TIME_WINDOW))
			ax.set_ylabel("Time Window")
			ax.set_xlabel("Id")
			ax.set_xticks(
				np.arange(.5, self.simulator.nb_resources, 1), minor=True)
			ax.set_yticks(np.arange(.5, GridEnv.TIME_WINDOW, 1), minor=True)
			ax.set_aspect('auto')
			ax.set_title("RES")
			ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

		def plot_job_state():
			jobs = self.simulator.get_job_slot_state()
			slot = 1
			for start_idx in range(0, jobs.shape[1], self.simulator.nb_resources):
				job_state = jobs[:, start_idx:start_idx + self.simulator.nb_resources]
				plt.subplot(1, 1 + GridEnv.JOB_SLOTS + 2, slot + 1)
				plt.imshow(job_state, interpolation='nearest', vmin=0, vmax=1, aspect='auto')
				ax = plt.gca()
				ax.set_xticks([], [])
				ax.set_yticks([], [])
				ax.set_xticks(
					np.arange(.5, self.simulator.nb_resources, 1), minor=True)
				ax.set_yticks(np.arange(.5, GridEnv.TIME_WINDOW, 1), minor=True)
				ax.set_title("Slot {}".format(slot))
				ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
				slot += 1

		def plot_backlog():
			backlog_state = self.simulator.get_backlog_state()
			plt.subplot(1, 1 + GridEnv.JOB_SLOTS + 2, GridEnv.JOB_SLOTS + 2)

			plt.imshow(backlog_state, interpolation='nearest', vmin=0, vmax=1, aspect='auto')
			ax = plt.gca()
			ax.set_xticks(range(GridEnv.BACKLOG_WIDTH))
			ax.set_yticks([], [])
			ax.set_xticks(np.arange(.5, GridEnv.BACKLOG_WIDTH, 1), minor=True)
			ax.set_yticks(np.arange(.5, GridEnv.TIME_WINDOW, 1), minor=True)
			ax.set_title("Queue")
			ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

		def plot_last_time():
			last_state = self.simulator.get_time_state().reshape(-1, 1)
			plt.subplot(1, 1 + GridEnv.JOB_SLOTS + 2, GridEnv.JOB_SLOTS + 3)

			plt.imshow(last_state, interpolation='nearest', vmin=0, vmax=1, aspect='auto')
			ax = plt.gca()
			ax.set_xticks([], [])
			ax.set_yticks([], [])
			ax.set_yticks(np.arange(.5, GridEnv.TIME_WINDOW, 1), minor=True)
			ax.set_title("Time")
			ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

		plt.figure("screen", figsize=(20, 5))
		plot_resource_state()
		plot_job_state()
		plot_backlog()
		plot_last_time()
		# plt.tight_layout()
		plt.pause(0.01)
