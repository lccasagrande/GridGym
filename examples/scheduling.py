import argparse

import gym
import gridgym.envs.off_reservation_env as e
from batsim_py.utils.graphics import plot_simulation_graphics


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
        obs, reward, done, info = env.step(1)
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
