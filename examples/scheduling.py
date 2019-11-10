import argparse
import math

import numpy as np
import gym
import gridgym.envs.off_reservation_env as e
from batsim_py.utils.graphics import plot_simulation_graphics



def act(obs):
    queue = [j for j in obs['queue']['jobs'] if j is not None]
    agenda = obs['agenda']
    nb_available = len(agenda) - np.count_nonzero(agenda)

    if queue:
        if queue[0]['res'] <= nb_available:
            return 1
        else:
            start_time = sorted(agenda)[:queue[0]['res']]
            start_time = math.ceil(start_time[-1])
            idx = sorted(range(1, len(queue)), key=lambda i: queue[i]['walltime'] * queue[i]['res'])
            for i in idx:
                if queue[i]['res'] <= nb_available and queue[i]['walltime'] <= start_time:
                    return i
    return 0

def run(args):
    print("[RUNNING]")

    env = gym.make(
        args.env_id,
        use_batsim=False,
        tax=86,
        timeout=5,
        act_interval=1,
        export=True,
        max_queue_sz=args.queue_sz)

    obs, done, score = env.reset(), False, 0
    while not done:
        obs, reward, done, info = env.step(act(obs))
        score += reward

    print("[DONE] Score: {} - Output: /tmp/GridGym/{}".format(score,
                                                              info['workload_name']))

    if args.plot_results:
        plot_simulation_graphics(
            "/tmp/GridGym/{}".format(info['workload_name']), show=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="Scheduling-v0", type=str)
    parser.add_argument("--plot_results", default=True, action="store_true")

    # Agent specific args
    parser.add_argument("--queue_sz", default=20, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
