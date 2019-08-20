import operator as op
from collections import defaultdict, OrderedDict
from enum import Enum, auto

import numpy as np
from procset import ProcSet


class PowerStateType(Enum):
    sleep = 0
    computation = 1
    switching_on = 2
    switching_off = 3

    def __str__(self):
        return self.name

    def __eq__(self, value):
        if isinstance(value, str):
            return self.name == value
        return super().__eq__(value)


class ResourceState(Enum):
    idle = 0
    switching_on = 1
    sleeping = 2
    switching_off = 3
    computing = 4

    def __str__(self):
        return self.name

    def __eq__(self, value):
        if isinstance(value, str):
            return self.name == value
        return super().__eq__(value)


class PowerState(object):
    def __init__(self, id, pstate_type, idle_power, comp_power):
        self.id = id
        self.type = pstate_type
        self.idle_power = idle_power
        self.comp_power = comp_power

    def __eq__(self, value):
        return self.id == value.id


class Resource(object):
    def __init__(self, id, parent_id, name, state, role, power_states):
        assert isinstance(state, ResourceState)
        self.power_states = sorted(power_states, key=lambda p: int(p.id))
        self.id = id
        self.parent_id = parent_id
        self.name = name
        self.state = state
        self.role = role
        self.pstate = next(
            ps for ps in self.power_states if ps.type == PowerStateType.computation
        )
        self.allocated_job = None

    @property
    def power(self):
        return (
            self.pstate.comp_power
            if self.state == ResourceState.computing
            else self.pstate.idle_power
        )

    @property
    def is_idle(self):
        return self.state == ResourceState.idle

    @property
    def is_computing(self):
        return self.state == ResourceState.computing

    @property
    def is_switching_off(self):
        return self.state == ResourceState.switching_off

    @property
    def is_switching_on(self):
        return self.state == ResourceState.switching_on

    @property
    def is_sleeping(self):
        return self.state == ResourceState.sleeping

    @property
    def is_reserved(self):
        return self.allocated_job != None

    def reserve(self, job):
        assert not self.is_reserved
        self.allocated_job = job

    def get_power_state(self, pstate_id):
        return self.power_states[int(pstate_id)]

    def execute(self, job):
        assert self.is_idle
        self.reserve(job)
        self.start_computing()

    def start_computing(self):
        assert self.is_reserved
        assert self.is_idle
        self.state = ResourceState.computing

    def release(self):
        self.allocated_job = None
        self.state = ResourceState.idle

    def sleep(self):
        assert self.is_idle
        pstate = next(
            p for p in self.power_states if p.type == PowerStateType.switching_off
        )
        self.state = ResourceState.switching_off
        self.pstate = pstate

    def wakeup(self):
        assert self.is_sleeping
        pstate = next(
            p for p in self.power_states if p.type == PowerStateType.switching_on
        )
        self.state = ResourceState.switching_on
        self.pstate = pstate

    def set_pstate(self, pstate_id):
        pstate = self.power_states[int(pstate_id)]
        if pstate.type == PowerStateType.sleep:
            assert self.is_switching_off
            self.state = ResourceState.sleeping
            self.pstate = pstate
        elif pstate.type == PowerStateType.computation:
            assert not self.is_switching_off and not self.is_sleeping
            if not self.is_computing:
                self.state = ResourceState.idle
            self.pstate = pstate


class Node:
    def __init__(self, id, resources):
        assert resources
        power_states = resources[0].power_states
        assert all(
            r.parent_id == id and r.power_states == power_states for r in resources
        )
        self.id = id
        self.resources = resources
        self.power_states = power_states
        self.nb_resources = len(resources)
        self.nb_state_switches = 0

    @property
    def state(self):
        return self.resources[0].pstate

    @property
    def power(self):
        return sum(r.power for r in self.resources)

    @property
    def is_idle(self):
        return self.is_on and all(not r.is_reserved for r in self.resources)

    @property
    def is_off(self):
        return self.state.type == PowerStateType.sleep

    @property
    def is_on(self):
        return self.state.type == PowerStateType.computation

    @property
    def is_switching_off(self):
        return self.state.type == PowerStateType.switching_off

    @property
    def is_switching_on(self):
        return self.state.type == PowerStateType.switching_on

    def get_power_state(self, pstate_id):
        return self.resources[0].get_power_state(pstate_id)

    def set_pstate(self, pstate_id):
        for r in self.resources:
            r.set_pstate(pstate_id)

    def sleep(self):
        for r in self.resources:
            r.sleep()
        self.nb_state_switches += 1

    def wakeup(self):
        for r in self.resources:
            r.wakeup()
        self.nb_state_switches += 1


class Platform:
    def __init__(self, nodes):
        assert len(nodes) > 0, "A platform must have at least one node"
        assert all(n.nb_resources == nodes[0].nb_resources for n in nodes[1:])
        self.__nodes, self.__resources = {}, {}
        for node in nodes:
            self.__nodes[node.id] = node
            self.__resources.update({r.id: r for r in node.resources})

        self.nb_nodes = len(nodes)
        self.nb_resources = len(self.__resources)

    @property
    def nodes(self):
        return list(self.__nodes.values())

    @property
    def resources(self):
        return list(self.__resources.values())

    def get_nodes(self, node_ids):
        n = op.itemgetter(*node_ids)(self.__nodes)
        return np.asarray(n) if isinstance(n, tuple) else np.asarray([n])

    def get_resources(self, resource_ids):
        r = op.itemgetter(*resource_ids)(self.__resources)
        return np.asarray(r) if isinstance(r, tuple) else np.asarray([r])

    def get_node(self, node_id):
        return self.__nodes[node_id]

    def get_resource(self, resource_id):
        return self.__resources[resource_id]

    def get_resources_state(self):
        return np.asarray([r.state for r in self.resources])
