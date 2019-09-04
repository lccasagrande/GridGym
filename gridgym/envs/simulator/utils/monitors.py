
import time as tm
from abc import abstractmethod
from copy import copy
from itertools import groupby

import pandas as pd
from procset import ProcSet

from gridgym.envs.simulator.job import Job, JobState
from gridgym.envs.simulator.resource import ResourceState, PowerStateType
from gridgym.envs.simulator.network import SimulationEventHandler


class SimulationMonitor(SimulationEventHandler):
    @abstractmethod
    def to_csv(self, fn):
        raise NotImplementedError


class JobMonitor(SimulationMonitor):
    def __init__(self, simulator):
        super().__init__(simulator)
        self._jobs_running = {}
        self.info = {
            'job_id': [],
            'workload_name': [],
            'profile': [],
            'submission_time': [],
            'requested_number_of_resources': [],
            'requested_time': [],
            'success': [],
            'final_state': [],
            'starting_time': [],
            'execution_time': [],
            'finish_time': [],
            'waiting_time': [],
            'turnaround_time': [],
            'stretch': [],
            'allocated_resources': [],
            'consumed_energy': [],
            'metadata': []
        }

    def to_csv(self, fn):
        pd.DataFrame.from_dict(self.info).to_csv(fn, index=False)

    def on_simulation_begins(self, timestamp, data):
        self._jobs_running = {}
        self.info = {k: [] for k in self.info.keys()}

    def on_simulation_ends(self, timestamp, data):
        self._jobs_running = {}

    def on_job_submitted(self, timestamp, data):
        self._jobs_running[data.job_id] = copy(data.job)

    def on_job_started(self, timestamp, data):
        job = self._jobs_running[data.job_id]
        job.set_allocation(data.alloc)
        job.start(timestamp)

    def on_job_killed(self, timestamp, data):
        for job_id in data.job_ids:
            job = self._jobs_running.pop(data.job_id)
            job.terminate(timestamp, JobState.COMPLETED_KILLED)
            self._update_info(job)

    def on_job_completed(self, timestamp, data):
        job = self._jobs_running.pop(data.job_id)
        job.terminate(timestamp, data.job_state)
        self._update_info(job)

    def _update_info(self, job):
        self.info['job_id'].append(job.id)
        self.info['workload_name'].append(job.workload_name)
        self.info['profile'].append(job.profile)
        self.info['submission_time'].append(job.subtime)
        self.info['requested_number_of_resources'].append(job.res)
        self.info['requested_time'].append(job.walltime)
        self.info['success'].append(
            0 if job.state == JobState.COMPLETED_KILLED else 1)
        self.info['final_state'].append(str(job.state))
        self.info['starting_time'].append(job.start_time)
        self.info['execution_time'].append(job.runtime)
        self.info['finish_time'].append(job.stop_time)
        self.info['waiting_time'].append(job.waiting_time)
        self.info['turnaround_time'].append(job.turnaround_time)
        self.info['stretch'].append(job.slowdown)
        self.info['allocated_resources'].append(str(ProcSet(*job.allocation)))
        self.info['consumed_energy'].append(-1)
        self.info['metadata'].append("")


class JobsStatsMonitor(SimulationMonitor):
    def __init__(self, simulator):
        super().__init__(simulator)
        self._jobs = {}
        self.info = {
            'makespan': 0,
            'max_slowdown': 0,
            'max_stretch': 0,
            'max_waiting_time': 0,
            'max_turnaround_time': 0,
            'mean_slowdown': 0,
            'mean_stretch': 0,
            'mean_waiting_time': 0,
            'mean_turnaround_time': 0,
            'nb_jobs': 0,
            'nb_jobs_finished': 0,
            'nb_jobs_killed': 0,
            'nb_jobs_success': 0,
        }

    def to_csv(self, fn):
        pd.DataFrame.from_dict(
            self.info, orient='index').T.to_csv(fn, index=False)

    def on_simulation_begins(self, timestamp, data):
        self._jobs = {}
        self.info = {k: 0 for k in self.info.keys()}

    def on_simulation_ends(self, timestamp, data):
        self.info['makespan'] = timestamp
        nb_finished = max(1, self.info['nb_jobs_finished'])
        self.info['mean_waiting_time'] /= nb_finished
        self.info['mean_slowdown'] /= nb_finished
        self.info['mean_stretch'] /= nb_finished
        self.info['mean_turnaround_time'] /= nb_finished

    def on_job_submitted(self, timestamp, data):
        self.info['nb_jobs'] += 1
        self._jobs[data.job_id] = copy(data.job)

    def on_job_started(self, timestamp, data):
        job = self._jobs[data.job_id]
        job.set_allocation(data.alloc)
        job.start(timestamp)

    def on_job_completed(self, timestamp, data):
        job = self._jobs[data.job_id]
        job.terminate(timestamp, data.job_state)
        self.info['makespan'] = timestamp

        self.info['mean_waiting_time'] += job.waiting_time
        self.info['mean_slowdown'] += job.slowdown
        self.info['mean_turnaround_time'] += job.turnaround_time
        self.info['mean_stretch'] += job.stretch

        self.info['max_waiting_time'] = max(
            job.waiting_time, self.info['max_waiting_time'])
        self.info['max_stretch'] = max(
            job.stretch, self.info['max_stretch'])
        self.info['max_slowdown'] = max(
            job.slowdown, self.info['max_slowdown'])
        self.info['max_turnaround_time'] = max(
            job.turnaround_time, self.info['max_turnaround_time'])

        self.info['nb_jobs_finished'] += 1
        if job.state == JobState.COMPLETED_SUCCESSFULLY:
            self.info['nb_jobs_success'] += 1
        elif job.state == JobState.COMPLETED_KILLED or job.state == JobState.COMPLETED_WALLTIME_REACHED:
            self.info['nb_jobs_killed'] += 1


class ResourceStatsMonitor(SimulationMonitor):
    def __init__(self, simulator):
        super().__init__(simulator)
        self.last_resource_state = self._platform = self._jobs_running = None
        self.info = {
            'time_idle':  0,
            'time_computing':  0,
            'time_switching_off':  0,
            'time_switching_on':  0,
            'time_sleeping':  0,
            'consumed_joules':  0,
            'energy_waste':  0,
            'nb_switches': 0,
        }

    def to_csv(self, fn):
        pd.DataFrame.from_dict(
            self.info, orient='index').T.to_csv(fn, index=False)

    def on_simulation_begins(self, timestamp, data):
        self._jobs_running = {}
        self._platform = data.platform
        self.last_resource_state = {
            r.id: (r.pstate, r.state, timestamp) for r in self._platform.resources}

        self.info = {k: 0 for k in self.info.keys()}

    def on_simulation_ends(self, timestamp, data):
        self._update_resource_state(
            timestamp, [r.id for r in self._platform.resources])
        self.last_resource_state = self._platform = self._jobs_running = None

    def on_job_started(self, timestamp, data):
        self._jobs_running[data.job_id] = data.alloc
        self._update_resource_state(
            timestamp, data.alloc, new_state=ResourceState.computing)

    def on_job_completed(self, timestamp, data):
        del self._jobs_running[data.job_id]
        self._update_resource_state(
            timestamp, data.alloc, new_state=ResourceState.idle)

    def on_job_killed(self, timestamp, data):
        resources = [
            a for job_id in data.job_ids for a in self._jobs_running.pop(job_id)
        ]
        self._update_resource_state(
            timestamp, resources, new_state=ResourceState.idle)

    def on_resource_power_state_changed(self, timestamp, data):
        self._update_resource_state(
            timestamp, data.resources, new_pstate_id=data.pstate)

    def _update_resource_state(self, timestamp, resources, new_state=None, new_pstate_id=None):
        nodes = []
        for r in self._platform.resources:
            r_pstate, r_state, r_tstart = self.last_resource_state[r.id]
            time_spent = timestamp - r_tstart

            if r_state == ResourceState.idle:
                self.info['time_idle'] += time_spent
                self.info['consumed_joules'] += r_pstate.power_min * time_spent
                self.info['energy_waste'] += r_pstate.power_min * time_spent
            elif r_state == ResourceState.computing:
                self.info['time_computing'] += time_spent
                self.info['consumed_joules'] += r_pstate.power_max * time_spent
            elif r_state == ResourceState.switching_off:
                self.info['time_switching_off'] += time_spent
                self.info['consumed_joules'] += r_pstate.power_min * time_spent
                self.info['energy_waste'] += r_pstate.power_min * time_spent
            elif r_state == ResourceState.switching_on:
                self.info['time_switching_on'] += time_spent
                self.info['consumed_joules'] += r_pstate.power_min * time_spent
                self.info['energy_waste'] += r_pstate.power_min * time_spent
            elif r_state == ResourceState.sleeping:
                self.info['time_sleeping'] += time_spent
                self.info['consumed_joules'] += r_pstate.power_min * time_spent
            else:
                raise NotImplementedError

            if r.id in resources:
                r_pstate = r_pstate if not new_pstate_id else next(
                    ps for ps in r.power_states if ps.id == new_pstate_id)

                if r_pstate.type == PowerStateType.sleep:
                    r_state = ResourceState.sleeping
                elif r_pstate.type == PowerStateType.switching_off:
                    r_state = ResourceState.switching_off
                    if new_pstate_id:
                        nodes.append(r.parent_id)
                elif r_pstate.type == PowerStateType.switching_on:
                    r_state = ResourceState.switching_on
                    if new_pstate_id:
                        nodes.append(r.parent_id)
                elif r_pstate.type == PowerStateType.computation:
                    if r_state == ResourceState.switching_on:
                        r_state = ResourceState.idle
                    if new_state is not None:
                        r_state = new_state
                else:
                    raise NotImplementedError

            self.last_resource_state[r.id] = (r_pstate, r_state, timestamp)

        self.info['nb_switches'] += len(set(nodes))


class SchedulerStatsMonitor(SimulationMonitor):
    def __init__(self, simulator):
        super().__init__(simulator)
        self.job_monitor = JobsStatsMonitor(simulator)
        self.resource_monitor = ResourceStatsMonitor(simulator)
        self.sim_start_time = self.simulation_time = -1

    @property
    def info(self):
        info = dict(self.job_monitor.info, **self.resource_monitor.info)
        info['simulation_time'] = self.simulation_time
        return info

    def on_simulation_begins(self, timestamp, data):
        self.sim_start_time = tm.time()

    def on_simulation_ends(self, timestamp, data):
        self.simulation_time = tm.time() - self.sim_start_time

    def to_csv(self, fn):
        pd.DataFrame.from_dict(
            self.info, orient='index').T.to_csv(fn, index=False)


class ResourceStatesEventMonitor(SimulationMonitor):
    def __init__(self, simulator):
        super().__init__(simulator)
        self.last_resource_state = self._platform = self._jobs_running = None
        self.info = {
            'time': [], 'nb_sleeping': [], 'nb_switching_on': [], 'nb_switching_off': [], 'nb_idle': [], 'nb_computing': []
        }

    def to_csv(self, fn):
        pd.DataFrame.from_dict(self.info).to_csv(fn, index=False)

    def on_simulation_begins(self, timestamp, data):
        self.info = {k: [] for k in self.info.keys()}
        self._jobs_running = {}
        self._platform = data.platform
        self.last_resource_state = {
            r.id: (r.pstate, r.state) for r in self._platform.resources}
        self._update_info(timestamp)

    def on_simulation_ends(self, timestamp, data):
        self._update_info(timestamp)
        self.last_resource_state = self._platform = self._jobs_running = None

    def on_job_started(self, timestamp, data):
        self._jobs_running[data.job_id] = data.alloc
        self._update_resource_state(
            timestamp, data.alloc, ResourceState.computing)

    def on_job_completed(self, timestamp, data):
        del self._jobs_running[data.job_id]
        self._update_resource_state(timestamp, data.alloc, ResourceState.idle)

    def on_job_killed(self, timestamp, data):
        resources = [
            a for job_id in data.job_ids for a in self._jobs_running.pop(job_id)
        ]
        self._update_resource_state(timestamp, resources, ResourceState.idle)

    def on_resource_power_state_changed(self, timestamp, data):
        self._update_resource_state(
            timestamp, data.resources, new_pstate_id=data.pstate)

    def _update_resource_state(self, timestamp, resources, new_state=None, new_pstate_id=None):
        for r in self._platform.get_resources(resources):
            r_pstate, r_state = self.last_resource_state[r.id]

            r_pstate = r_pstate if not new_pstate_id else next(
                ps for ps in r.power_states if ps.id == new_pstate_id)

            if r_pstate.type == PowerStateType.sleep:
                r_state = ResourceState.sleeping
            elif r_pstate.type == PowerStateType.switching_off:
                r_state = ResourceState.switching_off
            elif r_pstate.type == PowerStateType.switching_on:
                r_state = ResourceState.switching_on
            elif r_pstate.type == PowerStateType.computation:
                if r_state == ResourceState.switching_on:
                    r_state = ResourceState.idle
                if new_state is not None:
                    r_state = new_state
            else:
                raise NotImplementedError

            self.last_resource_state[r.id] = (r_pstate, r_state)
        self._update_info(timestamp)

    def _update_info(self, timestamp):
        nb_slp = nb_soff = nb_son = nb_cmp = nb_idl = 0
        for (pstate, state) in self.last_resource_state.values():
            if pstate.type == PowerStateType.sleep:
                nb_slp += 1
            elif pstate.type == PowerStateType.switching_off:
                nb_soff += 1
            elif pstate.type == PowerStateType.switching_on:
                nb_son += 1
            elif pstate.type == PowerStateType.computation:
                if state == ResourceState.computing:
                    nb_cmp += 1
                else:
                    nb_idl += 1

        if self.info['time'] and self.info['time'][-1] == timestamp:
            self.info['nb_sleeping'][-1] = nb_slp
            self.info['nb_switching_off'][-1] = nb_soff
            self.info['nb_switching_on'][-1] = nb_son
            self.info['nb_computing'][-1] = nb_cmp
            self.info['nb_idle'][-1] = nb_idl
        else:
            self.info['time'].append(timestamp)
            self.info['nb_sleeping'].append(nb_slp)
            self.info['nb_switching_off'].append(nb_soff)
            self.info['nb_switching_on'].append(nb_son)
            self.info['nb_computing'].append(nb_cmp)
            self.info['nb_idle'].append(nb_idl)


class ResourcePowerStatesEventMonitor(SimulationMonitor):
    def __init__(self, simulator):
        super().__init__(simulator)
        self._platform = None
        self.info = {'time': [], 'machine_id': [], 'new_pstate': []}

    def to_csv(self, fn):
        pd.DataFrame.from_dict(self.info).to_csv(fn, index=False)

    def on_simulation_begins(self, timestamp, data):
        self.info = {k: [] for k in self.info.keys()}
        self._platform = data.platform
        for ps_id, group in groupby(data.platform.resources, key=lambda r: r.pstate.id):
            self._update_info(timestamp, [r.id for r in group], ps_id)

    def on_simulation_ends(self, timestamp, data):
        self._platform = None

    def on_resource_power_state_changed(self, timestamp, data):
        self._update_info(timestamp, data.resources, data.pstate)

    def _update_info(self, timestamp, resources, new_pstate_id):
        r = self._platform.get_resource(resources[0])
        ps = next(ps for ps in r.power_states if ps.id == new_pstate_id)

        if ps.type == PowerStateType.switching_off:
            new_pstate = '-2'
        elif ps.type == PowerStateType.switching_on:
            new_pstate = '-1'
        else:
            new_pstate = ps.id

        self.info['time'].append(timestamp)
        self.info['machine_id'].append(str(ProcSet(*resources)))
        self.info['new_pstate'].append(new_pstate)


class EnergyEventMonitor(SimulationMonitor):
    def __init__(self, simulator):
        super().__init__(simulator)
        self.last_resource_state = self._platform = self._jobs_running = None
        self.info = {
            'time': [], 'energy': [], 'event_type': [], 'wattmin': [], 'epower': []
        }

    def to_csv(self, fn):
        pd.DataFrame.from_dict(self.info).to_csv(fn, index=False)

    def on_simulation_begins(self, timestamp, data):
        self._jobs_running = {}
        self._platform = data.platform
        self.last_resource_state = {
            r.id: (r.pstate, r.state, timestamp) for r in self._platform.resources}

        self.info = {k: [] for k in self.info.keys()}

    def on_simulation_ends(self, timestamp, data):
        self.last_resource_state = self._platform = self._jobs_running = None

    def on_job_started(self, timestamp, data):
        self._jobs_running[data.job_id] = data.alloc
        self._update_resource_state(
            timestamp, data.alloc, event_type='s', new_state=ResourceState.computing)

    def on_job_completed(self, timestamp, data):
        del self._jobs_running[data.job_id]
        self._update_resource_state(
            timestamp, data.alloc, event_type='e', new_state=ResourceState.idle)

    def on_job_killed(self, timestamp, data):
        resources = [
            a for job_id in data.job_ids for a in self._jobs_running.pop(job_id)
        ]
        self._update_resource_state(
            timestamp, resources, event_type='e', new_state=ResourceState.idle)

    def on_resource_power_state_changed(self, timestamp, data):
        self._update_resource_state(
            timestamp, data.resources, event_type='p', new_pstate_id=data.pstate)

    def _update_resource_state(self, timestamp, resources, event_type, new_state=None, new_pstate_id=None):
        energy = self.info['energy'][-1] if self.info['energy'] else 0
        epower = 0
        for r in self._platform.resources:
            r_pstate, r_state, r_tstart = self.last_resource_state[r.id]
            time_spent = timestamp - r_tstart

            if r_state == ResourceState.idle:
                epower += r_pstate.power_min
                energy += r_pstate.power_min * time_spent
            elif r_state == ResourceState.computing:
                epower += r_pstate.power_max
                energy += r_pstate.power_max * time_spent
            elif r_state == ResourceState.switching_off:
                epower += r_pstate.power_min
                energy += r_pstate.power_min * time_spent
            elif r_state == ResourceState.switching_on:
                epower += r_pstate.power_min
                energy += r_pstate.power_min * time_spent
            elif r_state == ResourceState.sleeping:
                epower += r_pstate.power_min
                energy += r_pstate.power_min * time_spent
            else:
                raise NotImplementedError

            if r.id in resources:
                r_pstate = r_pstate if not new_pstate_id else next(
                    ps for ps in r.power_states if ps.id == new_pstate_id)
                if r_pstate.type == PowerStateType.sleep:
                    r_state = ResourceState.sleeping
                elif r_pstate.type == PowerStateType.switching_off:
                    r_state = ResourceState.switching_off
                elif r_pstate.type == PowerStateType.switching_on:
                    r_state = ResourceState.switching_on
                elif r_pstate.type == PowerStateType.computation:
                    if r_state == ResourceState.switching_on:
                        r_state = ResourceState.idle
                    if new_state is not None:
                        r_state = new_state
                else:
                    raise NotImplementedError

            self.last_resource_state[r.id] = (r_pstate, r_state, timestamp)
        self.info['time'].append(timestamp)
        self.info['energy'].append(energy)
        self.info['event_type'].append(event_type)
        self.info['wattmin'].append(-1)
        self.info['epower'].append(epower)
