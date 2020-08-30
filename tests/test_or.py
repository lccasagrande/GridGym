import pytest

import gym
import numpy as np
import batsim_py
from batsim_py.resources import Platform, Host, PowerState, PowerStateType

from gridgym.envs.off_reservation_env import OffReservationEnv


def get_platform(nb_hosts):
    pstates = [
        PowerState(0, PowerStateType.COMPUTATION, 90, 180),
        PowerState(1, PowerStateType.SLEEP, 9, 9),
        PowerState(2, PowerStateType.SWITCHING_OFF, 120, 120),
        PowerState(3, PowerStateType.SWITCHING_ON, 100, 100),
    ]

    hosts = [Host(i, str(i), pstates, True) for i in range(nb_hosts)]
    return Platform(hosts)


class TestOffReservationEnv:
    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        mocker.patch("os.path.exists", return_value=True)
        mocker.patch("os.listdir", return_value=["w.json"])
        mocker.patch.object(batsim_py.SimulatorHandler, "start")
        mocker.patch.object(batsim_py.SimulatorHandler,
                            "platform",
                            new_callable=mocker.PropertyMock,
                            return_value=get_platform(2))

    def test_platform_status_shape(self):
        env = OffReservationEnv("p.xml", "ps/")
        p = env.observation_space['platform']['status']
        assert p.shape == (2, 1)

    def test_platform_status_shape_with_multiple_servers(self, mocker):
        mocker.patch.object(batsim_py.SimulatorHandler,
                            "platform",
                            new_callable=mocker.PropertyMock,
                            return_value=get_platform(18))

        env = OffReservationEnv("p.xml", "ps/", hosts_per_server=6)
        p = env.observation_space['platform']['status']
        assert p.shape == (3, 6)

