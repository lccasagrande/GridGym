import argparse

import gym

import gridgym.envs.off_reservation_env as e


class FirstFitScheduler():
    def act(self, obs):
        agenda = obs['platform']['agenda']
        queue = obs['queue']['jobs']
        nb_available = len(agenda) - sum(1 for j in agenda if j[1] != 0)
        job_pos = next((i for i, j in enumerate(queue)
                        if 0 < j[1] <= nb_available), -1)
        return job_pos + 1

    def play(self, env, verbose=True):
        history = {"score": 0, 'steps': 0, 'info': None}
        obs, done, info = env.reset(), False, {}
        while not done:
            obs, reward, done, info = env.step(self.act(obs))
            history['score'] += reward
            history['steps'] += 1
            history['info'] = info

        if verbose:
            print(
                "[DONE] Score: {} - Steps: {}".format(history['score'], history['steps']))
        env.close()
        return history


def run(args):
    print("[RUNNING]")

    agent = FirstFitScheduler()

    env = gym.make(args.env_id,
                   platform_fn="files/platforms/platform.xml",
                   workloads_dir="files/workloads/",
                   t_action=args.t_action,
                   queue_max_len=args.queue_sz,
                   t_shutdown=5,
                   hosts_per_server=12)

    hist = agent.play(env, True)

    print("[DONE]")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="gridgym:Scheduling-v0", type=str)
    # Agent specific args
    parser.add_argument("--queue_sz", default=20, type=int)
    parser.add_argument("--t_action", default=1, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
