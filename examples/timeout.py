import argparse

import numpy as np
import gym
from gym import spaces
import gridgym.envs.off_reservation_env as e
from batsim_py.utils.graphics import plot_simulation_graphics


class TimeoutPolicy():
    def __init__(self, timeout, nb_resources, nb_nodes, queue_sz):
        self.timeout = timeout
        self.nb_resources = nb_resources
        self.nodes_state = np.full(shape=nb_nodes, fill_value=-1, dtype=np.int)
        self.current_time = 0
        self.queue_sz = queue_sz

    def _get_available_resources(self, obs):
        nb_available = sum(1 if a == 0 else 0 for a in obs['agenda'])
        queue = obs['queue'][:self.queue_sz]
        if len(queue) > 0:
            (sub, res, wall, pjob_expected_t_start, user, profile) = queue[0]
            if pjob_expected_t_start == 0 or res <= nb_available:
                nb_available -= res
                pjob_expected_t_start = -1
                queue = queue[1:]
            for (sub, res, wall, expected_t_start, user, profile) in queue:
                if pjob_expected_t_start == -1 and res <= nb_available:
                    nb_available -= res
                elif res <= nb_available and wall <= pjob_expected_t_start:
                    nb_available -= res
        return nb_available

    def act(self, obs):
        if self.timeout == -1:
            return 0
        reservation_size, sim_time, platform = 0, obs['time'], obs['platform']
        nb_available = self._get_available_resources(obs)

        for i, node in enumerate(platform):
            if nb_available >= len(node):
                if all(r == 0 for r in node):
                    self.nodes_state[i] += sim_time - self.current_time
                else:
                    self.nodes_state[i] = -1
                if self.nodes_state[i] >= self.timeout or all(r == 2 or r == 3 for r in node):
                    reservation_size += 1
                    nb_available -= len(node)

        self.current_time = sim_time
        return reservation_size

    def play(self, env):
        self.nodes_state = np.full(
            shape=self.nodes_state.shape, fill_value=-1, dtype=np.int)
        self.current_time = 0
        obs, done, score = env.reset(), False, 0
        while not done:
            obs, reward, done, info = env.step(self.act(obs))
            score += reward
        env.close()
        return score, info


def run(args):
    print("[RUNNING]")

    env = gym.make(args.env_id, export=True)
    nb_resources = env.observation_space.spaces['agenda'].shape[0]
    nb_nodes = env.observation_space.spaces['platform'].shape[0]

    agent = TimeoutPolicy(args.timeout, nb_resources, nb_nodes, args.queue_sz)

    score, info = agent.play(env)

    print("[DONE] Score: {} - Output: /tmp/GridGym/{}".format(score,
                                                              info['workload_name']))

    if args.plot_results:
        plot_simulation_graphics(
            "/tmp/GridGym/{}".format(info['workload_name']), show=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="OffReservation-v0", type=str)
    parser.add_argument("--plot_results", default=True, action="store_true")

    # Agent specific args
    parser.add_argument("--timeout", default=0, type=int)
    parser.add_argument("--queue_sz", default=10, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
