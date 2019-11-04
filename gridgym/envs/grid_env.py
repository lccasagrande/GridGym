import os

import gym
from gym import error
from gym.utils import seeding

try:
    import batsim_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install batsim_py, see: https://github.com/lccasagrande/batsim-py/.)".format(e))

from batsim_py.rjms import RJMSHandler
from batsim_py.utils.monitors import *
from batsim_py.utils.commons import *
from batsim_py.utils.submitter import JobSubmitter


class GridEnv(gym.Env):
    OUTPUT = "/tmp/GridGym/"

    def __init__(self, use_batsim=False, simulation_time=None, files_dir=None, export=False, qos_stretch=None):
        self.seed()
        self.export = export
        self.simulation_time = simulation_time
        self.workloads = self.platform_fn = None
        self.qos_stretch = qos_stretch
        self.workload_name = ""
        self.rjms = self._get_rjms(use_batsim)
        self._load(files_dir)
        self.observation_space, self.action_space = self._get_space()
        self.metadata = {'render.modes': []}
        os.makedirs(self.OUTPUT, exist_ok=True)

    def _get_rjms(self, use_batsim):
        return RJMSHandler(use_batsim)


    def _load(self, files_dir):
        if files_dir is None:
            files_dir = os.path.join(os.path.dirname(__file__), 'files')

        if not os.path.exists(files_dir):
            raise IOError("Files in {} does not exist".format(files_dir))

        platform_fn = "{}/platform.xml".format(files_dir)
        if not os.path.exists(platform_fn):
            raise IOError("File {} does not exist".format(platform_fn))

        workloads_path = "{}/workloads/".format(files_dir)
        if not os.path.exists(workloads_path):
            raise IOError("Workloads {} does not exist".format(workloads_path))

        workloads = []
        for f in os.scandir(workloads_path):
            if f.is_dir():
                group_workloads = [
                    os.path.join(f, w) for w in os.listdir(f) if w.endswith('.json')
                ]
                if len(group_workloads) > 0:
                    workloads.append(group_workloads)
        if len(workloads) == 0:
            workloads = [
                os.path.join(workloads_path, w) for w in os.listdir(workloads_path) if w.endswith('.json')
            ]
        elif len(workloads) == 1:
            workloads = workloads[0]

        if not workloads:
            raise IOError("Workloads not found in {}".format(workloads_path))

        self.workloads = workloads
        self.platform_fn = platform_fn

    def _start_simulation(self):
        # self.np_random.shuffle(self.workloads)
        group = self.np_random.choice(self.workloads)
        workload = self.np_random.choice(group) if isinstance(group, list) else group
        self.workload_name = "{}".format(
            workload[workload.rfind('/')+1:workload.rfind('.json')])
        export_fn = self.OUTPUT + self.workload_name if self.export else None

        self.rjms.start(platform_fn=self.platform_fn,
                        workload_fn=workload,
                        simulation_time=self.simulation_time,
                        output_fn=export_fn,
                        qos_stretch=self.qos_stretch)

    def close(self, s=None, a=None):
        self.rjms.close()
        self.workloads = self.platform_fn = None

    def reset(self):
        assert len(self.workloads) > 0
        self.rjms.close()
        self._start_simulation()
        return self._get_obs()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError

    def _get_space(self):
        raise NotImplementedError

    def _get_obs(self):
        raise NotImplementedError
