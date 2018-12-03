import gym
import gridgym.envs.grid_env
import numpy as np
import time as tm
import pandas as pd
import random
import argparse

NB_RESOURCES = 10
NB_JOB_SLOTS = 10


class SJFAgent:
	def act(self, state):
		def get_avail_resources():
			return NB_RESOURCES - np.count_nonzero(state[0:NB_RESOURCES])

		def get_jobs():
			slot, jobs, jobs_state = 1, [], state[NB_RESOURCES:NB_RESOURCES + NB_JOB_SLOTS * 2]
			for i in range(0, NB_JOB_SLOTS * 2, 2):
				if jobs_state[i] != 0:  # There is a job in the job slot
					res = jobs_state[i]
					time = np.sum(jobs_state[i + 1])
					jobs.append((res, time, slot))
				slot += 1
			return jobs

		action, shortest_job = 0, np.inf
		avail_resources = get_avail_resources()
		jobs = get_jobs()
		for j in jobs:
			if j[0] <= avail_resources and j[1] < shortest_job:
				shortest_job = j[1]
				action = j[2]

		return action

	def evaluate(self, env, visualize=False, verbose=False):
		start_time = tm.time()
		ob = env.reset()
		reward, steps, done, info = 0.0, 0, False, {}
		while not done:
			if visualize:
				env.render()
			action = self.act(ob)
			ob, r, done, info = env.step(action)
			reward += r
			steps += 1
		if verbose:
			print("[EVALUATE][INFO] {}".format(" ".join([" [{}: {}]".format(k, v) for k, v in info.items()])))

		env.close()
		end_time = tm.time() - start_time
		print("[EVALUATE] Avg. reward {:.4f} - Avg. steps {:.4f} - Done in {:.4f} sec".format(reward, steps, end_time))
		return info


def run(args):
	agent = SJFAgent()
	env = gym.make(args.env)
	env.seed(args.seed)
	results = agent.evaluate(env, args.visualize, args.verbose)
	if args.output_fn is not None and results:
		pd.DataFrame([results]).to_csv(args.output, index=False)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", type=str, default="grid-v0")  # grid-v0
	parser.add_argument("--seed", default=123, type=int)
	parser.add_argument("--verbose", default=True, action="store_true")
	parser.add_argument("--output_fn", type=str, default=None)
	parser.add_argument("--visualize", default=False, action="store_true")
	return parser.parse_args()


if __name__ == "__main__":
	run(parse_args())
