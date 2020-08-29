import argparse
import math
from typing import Optional
from typing import Sequence
from typing import Dict
from typing import List

from batsim_py.resources import HostState
import gym
import numpy as np

import gridgym.envs.off_reservation_env as e


class TimeoutPolicy():
    def __init__(self, timeout: Optional[int] = None):
        self.timeout = timeout
        self.idle_servers: Dict[int, float] = {}

    def _count_available_servers(self, obs) -> int:
        agenda = []
        for t_start, t_wall, _ in obs['platform']['agenda']:
            if t_wall > 0:
                agenda.append(t_wall - (obs['current_time'] - t_start))
            elif t_wall == 0:
                agenda.append(0)
            else:
                agenda.append(np.inf)

        reserved: List[int] = []
        p_start_t: Optional[int] = None
        for (sub, res, wall, user_id) in obs['queue']['jobs']:
            if res == 0:
                break
            h_available = [i for i, r in enumerate(agenda) if r == 0]
            h_not_reserved = [i for i in h_available if i not in reserved]

            if res <= len(h_not_reserved):
                for alloc_i in h_not_reserved[:res]:
                    agenda[alloc_i] = wall
            elif p_start_t is None or not reserved:
                next_releases = sorted(enumerate(agenda), key=lambda a: a[1])
                last = min(len(next_releases), res) - 1
                p_start_t = next_releases[last][1]

                candidates = [
                    r[0] for r in next_releases if r[1] <= p_start_t
                ]
                reserved = candidates[-res:]
                if not h_available:
                    break
            elif wall > 0 and wall <= p_start_t and res <= len(h_available):
                for alloc_i in h_available[:res]:
                    agenda[alloc_i] = wall

        nb_avail = sum([1 for r in agenda if r == 0])
        servers_avail = nb_avail / obs['platform']['status'].shape[1]
        return math.ceil(servers_avail)

    def act(self, obs: dict) -> int:
        reservation_size = 0

        if self.timeout is None:
            return reservation_size

        nb_avail = self._count_available_servers(obs)
        if nb_avail == 0:
            return reservation_size

        reserved_states = (
            HostState.SLEEPING.value,
            HostState.SWITCHING_OFF.value
        )

        for i, server in enumerate(obs['platform']['status']):
            if all(h == HostState.IDLE.value for h in server):
                if i not in self.idle_servers:
                    self.idle_servers[i] = obs['current_time']
            else:
                self.idle_servers.pop(i, None)
                if any(h in reserved_states for h in server):
                    reservation_size += 1

        for server, t_start_idle in self.idle_servers.items():
            if obs['current_time'] - t_start_idle >= self.timeout:
                reservation_size += 1

        return min(nb_avail, reservation_size)

    def play(self, env, verbose=True):
        history = {"score": 0, 'steps': 0, 'info': None}
        obs, done, info = env.reset(), False, {}
        while not done:
            obs, reward, done, info = env.step(self.act(obs))
            history['score'] += reward
            history['steps'] += 1
            history['info'] = info

        if verbose:
            print("[DONE] Score: {} - Steps: {} - Output: /tmp/GridGym/{}".format(
                history['score'], history['steps'], info['workload']))
        env.close()
        return history


def run(args):
    print("[RUNNING]")
    env = gym.make(args.env_id,
                   platform_fn="files/platforms/platform.xml",
                   workloads_dir="files/workloads/",
                   t_action=args.t_action,
                   queue_max_len=args.queue_sz,
                   hosts_per_server=args.server_size,
                   qos_treshold=0.5,
                   simulation_time=args.sim_t)

    agent = TimeoutPolicy(args.t_timeout)

    agent.play(env, True)

    print("[DONE]")


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_id", default="gridgym:OffReservation-v0", type=str)
    # Agent specific args
    parser.add_argument("--queue_sz", default=10, type=int)
    parser.add_argument("--t_timeout", default=5, type=int)
    parser.add_argument("--t_action", default=1, type=int)
    parser.add_argument("--server_size", default=12, type=int)
    parser.add_argument("--sim_t", default=1440, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
