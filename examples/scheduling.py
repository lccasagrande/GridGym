import argparse
import math
import csv
import os.path as osp
from collections import defaultdict

import pandas as pd
import numpy as np
import gym

import gridgym.envs.off_reservation_env as e
from batsim_py.resource import ResourceState
from batsim_py.utils.graphics import plot_simulation_graphics


class FirstFitScheduler():
    def act(self, obs):
        queue = [j for j in obs['queue']['jobs'] if j is not None]
        nb_available = len(obs['agenda']) - np.count_nonzero(obs['agenda'])

        for i in range(len(queue)):
            if queue[i]['res'] <= nb_available:
                return i + 1
        return 0

    def play(self, env, verbose=True):
        history = {"score": 0, 'steps': 0, 'info': None}
        obs, done = env.reset(), False
        while not done:
            obs, reward, done, info = env.step(self.act(obs))
            history['score'] += reward
            history['steps'] += 1
            history['info'] = info

        if verbose:
            print("[DONE] Score: {} - Steps: {} - Output: /tmp/GridGym/{}".format(history['score'], history['steps'],
                                                                                  info['workload_name']))
        env.close()
        return history


def run(args):
    print("[RUNNING]")

    agent = FirstFitScheduler()

    env = gym.make(
        args.env_id,
        use_batsim=False,
        timeout=5,
        act_interval=1,
        export=True,
        max_queue_sz=args.queue_sz,
        max_simulation_time=args.sim_time)

    hist = agent.play(env, True)

    if args.plot_results:
        plot_simulation_graphics(
            "/tmp/GridGym/{}".format(hist['info']['workload_name']), show=True)

    print("[DONE]")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="Scheduling-v0", type=str)
    parser.add_argument("--plot_results", default=1, action="store_true")

    # Agent specific args
    parser.add_argument("--queue_sz", default=20, type=int)
    parser.add_argument("--sim_time", default=10*1440, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
