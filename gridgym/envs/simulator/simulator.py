from xml.dom import minidom
from itertools import takewhile

from gridgym.envs.simulator.resource import Platform, PowerStateType, ResourceState
from gridgym.envs.simulator.job import Job, JobState
from gridgym.envs.simulator.network import *
from gridgym.envs.simulator.utils.submitter import Workload, JobSubmitter
import subprocess


def get_free_tcp_address():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", 0))
    host, port = tcp.getsockname()
    tcp.close()
    return "tcp://127.0.0.1:{}".format(port)


def get_resources_from_platform(fn):
    platform = minidom.parse(fn)
    prop = platform.getElementsByTagName("prop")[0]
    assert prop.getAttribute("id") == 'cores_per_node'
    cores_per_node = int(prop.getAttribute("value"))
    hosts = platform.getElementsByTagName('host')
    hosts.sort(key=lambda x: x.attributes['id'].value)
    resources, id = [], 0
    for r in hosts:
        if r.getAttribute('id') != 'master_host':
            properties = {
                p.getAttribute('id'): p.getAttribute('value') for p in r.getElementsByTagName('prop')
            }
            properties['role'] = properties.get('role', '')
            resource = {
                'id': id,
                'name': r.getAttribute('id'),
                'pstate': r.getAttribute('pstate'),
                'speed': r.getAttribute('speed'),
                'properties': properties,
                'zone_properties': {'cores_per_node': cores_per_node}
            }
            resources.append(resource)
            id += 1
    return resources


class GridSimulationHandler(SimulationProtocol):
    def __init__(self):
        self.__current_time = 0.0
        self.__callbacks = {k: [] for k in EventType}
        self.__events = SortedList(key=lambda e: e.timestamp)
        self.__profiles = {}
        self.__submitter_ended = True
        self.__submitter = None
        self.__platform = None
        self._is_running = False
        self.__jobs = {}
        self.set_callback(EventType.SIMULATION_ENDS, self.on_simulation_ends)


    @property
    def address(self):
        return None

    @property
    def is_running(self):
        return self._is_running

    @property
    def current_time(self):
        return self.__current_time

    def ack(self):
        pass

    def set_callback(self, event_type, call):
        self.__callbacks[event_type].append(call)

    def proceed_simulation(self):
        assert self.is_running
        if len(self.__events) == 0 and self.__submitter_ended and len(self.__jobs) == 0:
            self.finish()
        elif len(self.__events) > 0:
            self.__current_time = self.__events[0].timestamp
            self._dispatch_events()
        else:
            raise RuntimeError

    def start(self, platform_fn, workload_fn=None, output_dir=None):
        assert not self.is_running
        assert platform_fn

        self.__submitter_ended = False
        self.__jobs = {}
        self.__current_time = 0.0

        resources = get_resources_from_platform(platform_fn)
        event = SimulationBeginsEvent(self.current_time, resources)
        self.__platform = event.data.platform

        if workload_fn:
            if self.__submitter is None:
                self.__submitter = JobSubmitter(self)
            self.__submitter.start(workload_fn)

        self.__events.add(event)
        self._is_running = True
        self._dispatch_events()

    def finish(self):
        self._is_running = False
        self.__events.add(SimulationEndsEvent(self.current_time))
        self._dispatch_events()

    def on_simulation_ends(self, timestamp, data):
        self.__submitter_ended = True
        if self.__submitter is not None:
            self.__submitter.close()
        self.__events.clear()
        self.__profiles.clear()
        self.__jobs = {}

    def execute_job(self, job_id, alloc):
        allocation = str(ProcSet(*alloc))
        event_1 = JobStartedEvent(self.current_time, job_id, allocation)

        job = self.__jobs.pop(job_id)
        min_speed = min(r.speed for r in self.__platform.get_resources(alloc))
        if job.profile not in self.__profiles:
            print(self.__profiles)
        job_profile = self.__profiles[job.profile]
        if job_profile.type == WorkloadProfileType.parallel_homogeneous:
            runtime = int(job_profile.cpu / min_speed)
        elif job_profile.type == WorkloadProfileType.parallel_homogeneous_total:
            cpu = job_profile.cpu / job.res
            runtime = int(cpu / min_speed)
        else:
            raise NotImplementedError
        event_2 = JobCompletedEvent(
            self.current_time + min(runtime, job.walltime),
            job_id,
            str(JobState.COMPLETED_SUCCESSFULLY if job.walltime >
                runtime else JobState.COMPLETED_WALLTIME_REACHED),
            "0",
            allocation
        )
        self.__events.add(event_1)
        self.__events.add(event_2)

    def reject_job(self, job_id):
        del self.__jobs[job_id]

    def call_me_later(self, at):
        assert at >= self.current_time
        events = takewhile(lambda e: e.timestamp <= int(at), self.__events)
        if not any(e.type == EventType.REQUESTED_CALL and e.timestamp == int(at) for e in events):
            self.__events.add(RequestedCallEvent(int(at)))

    def kill_job(self, job_ids):
        for i in job_ids:
            p = next(p for p, e in enumerate(self.__events)
                     if e.type == EventType.JOB_COMPLETED and e.data.job_id == i)
            self.__events.pop(p)
        self.__events.add(JobKilledEvent(self.current_time, job_ids))

    def register_job(self, id, profile, res, walltime):
        e = JobSubmittedEvent(self.current_time, id, profile, res, walltime)
        self.__jobs[id] = e.data.job
        self.__events.add(e)

    def register_profile(self, workload_name, profile_name, profile):
        assert isinstance(profile, WorkloadProfile)
        self.__profiles[profile_name] = profile

    def set_resources_pstate(self, resources, pstate):
        timestamps = defaultdict(list)
        transitions = defaultdict(list)
        nodes_visited = {}
        for resource in self.__platform.get_resources(resources):
            if resource.parent_id not in nodes_visited:
                n = self.__platform.get_node(resource.parent_id)
                next_ps = next(ps for ps in n.power_states if ps.id == pstate)
                if next_ps.type == PowerStateType.sleep:
                    trans_ps = next(ps for ps in n.power_states if ps.type ==
                                    PowerStateType.switching_off)
                elif next_ps.type == PowerStateType.computation and not n.is_on:
                    trans_ps = next(ps for ps in n.power_states if ps.type ==
                                    PowerStateType.switching_on)
                else:
                    trans_ps = None

                time_to_switch = 0 if not trans_ps else int(1/trans_ps.speed)
                for r in n.resources:
                    timestamps[time_to_switch].append(r.id)
                    if trans_ps:
                        transitions[trans_ps.id].append(r.id)
                nodes_visited[r.parent_id] = True

        for ps_id, res_ids in transitions.items():
            self.__events.add(
                ResourcePowerStateChangedEvent(
                    self.current_time, str(ProcSet(*res_ids)), ps_id
                )
            )

        for time_to_switch, ids in timestamps.items():
            e = ResourcePowerStateChangedEvent(
                self.current_time + time_to_switch, str(ProcSet(*ids)), pstate)
            self.__events.add(e)

    def change_job_state(self, job_id, job_state, kill_reason):
        raise NotImplementedError

    def notify(self, notify_type):
        if notify_type == NotifyType.no_more_static_job_to_submit:
            self.__submitter_ended = True
        self.__events.add(
            Notify(self.current_time, NotifyType[notify_type]))

    def _dispatch_events(self):
        while self.__events and self.__events[0].timestamp == self.current_time:
            event = self.__events.pop(0)
            for callback in self.__callbacks[event.type]:
                callback(event.timestamp, event.data)


class BatsimSimulationHandler(SimulationProtocol):
    WORKLOAD_JOB_SEPARATOR = "!"
    ATTEMPT_JOB_SEPARATOR = "#"
    WORKLOAD_JOB_SEPARATOR_REPLACEMENT = "%"

    def __init__(self, address=None):
        if address is None:
            address = get_free_tcp_address()
        self.__network = NetworkHandler(address)
        self.__requests = SortedList([], key=lambda r: r.timestamp)
        self.__current_time = 0.0
        self.__callbacks = {k: [] for k in EventType}
        self.__simulator = None
        self.__platform = None
        self.set_callback(EventType.SIMULATION_ENDS, self.on_simulation_ends)

    @property
    def address(self):
        return self.__network.address

    @property
    def is_running(self):
        return self.__simulator is not None

    @property
    def current_time(self):
        return self.__current_time

    def ack(self):
        self.__network.send(Message(self.current_time, []))

    def set_callback(self, event_type, call):
        self.__callbacks[event_type].append(call)

    def proceed_simulation(self):
        assert self.is_running
        self._dispatch_events(self._send_and_recv())

    def start(self, platform_fn, workload_fn=None, output_dir=None):
        assert not self.is_running
        assert platform_fn
        cmd = "batsim -s {} -p {} -E".format(self.address, platform_fn)
        cmd += ' -w {}'.format(workload_fn) if workload_fn else ""
        cmd += " -e {}".format(output_dir) if output_dir else ""

        self.__simulator = subprocess.Popen(
            cmd.split(), stdout=subprocess.PIPE, shell=False
        )
        self.__network.bind()
        self.__current_time = 0.0

        # Load platform from file instead of batsim's protocol message
        resources = get_resources_from_platform(platform_fn)
        event = SimulationBeginsEvent(self.current_time, resources)
        self.__platform = event.data.platform
        self._dispatch_events([event])
        self.__network.flush(blocking=True)

    def finish(self):
        if self.__simulator is not None:
            self.__simulator.terminate()
            self.__simulator = None

        self._dispatch_events([SimulationEndsEvent(self.current_time)])

    def on_simulation_ends(self, timestamp, data):
        if self.__simulator:
            self.ack()
            self.__simulator.wait()
            self.__simulator.terminate()
            self.__simulator = None
        self.__requests.clear()
        self.__network.close()

    def _dispatch_events(self, events):
        for e in events:
            for callback in self.__callbacks[e.type]:
                callback(e.timestamp, e.data)

    def execute_job(self, job_id, alloc):
        request = ExecuteJobRequest(self.current_time, job_id, alloc)
        self._dispatch_events([JobStartedEvent(
            self.current_time, job_id, request.data.alloc
        )])

        self._append_request(request)

    def reject_job(self, job_id):
        request = RejectJobRequest(self.current_time, job_id)
        self._append_request(request)

    def call_me_later(self, when):
        request = CallMeLaterRequest(self.current_time, when)
        self._append_request(request)

    def kill_job(self, job_ids):
        request = KillJobRequest(self.current_time, job_ids)
        self._append_request(request)

    def register_job(self, id, profile, res, walltime):
        request = RegisterJobRequest(
            self.current_time, id, profile, res, walltime)
        self._append_request(request)

    def register_profile(self, workload_name, profile_name, profile):
        request = RegisterProfileRequest(
            self.current_time, workload_name, profile_name, profile
        )
        self._append_request(request)

    def set_resources_pstate(self, resources, pstate):
        request = SetResourceStateRequest(self.current_time, resources, pstate)
        self._append_request(request)

        transitions = defaultdict(list)
        nodes_visited = {}
        for resource in self.__platform.get_resources(resources):
            if resource.parent_id not in nodes_visited:
                n = self.__platform.get_node(resource.parent_id)
                next_ps = next(ps for ps in n.power_states if ps.id == pstate)
                if next_ps.type == PowerStateType.sleep:
                    trans_ps = next(ps for ps in n.power_states if ps.type ==
                                    PowerStateType.switching_off)
                elif next_ps.type == PowerStateType.computation and not n.is_on:
                    trans_ps = next(ps for ps in n.power_states if ps.type ==
                                    PowerStateType.switching_on)
                else:
                    trans_ps = None
                if trans_ps:
                    for r in n.resources:
                        transitions[trans_ps.id].append(r.id)
                nodes_visited[r.parent_id] = True

        events = [
            ResourcePowerStateChangedEvent(
                self.current_time, str(ProcSet(*res_ids)), ps_id)
            for ps_id, res_ids in transitions.items()
        ]
        self._dispatch_events(events)

    def change_job_state(self, job_id, job_state, kill_reason):
        request = ChangeJobStateRequest(
            self.current_time, job_id, job_state, kill_reason
        )
        self._append_request(request)

    def notify(self, notify_type):
        request = Notify(self.current_time, NotifyType[notify_type])
        self._append_request(request)

    def _append_request(self, request):
        assert isinstance(request, Request) or isinstance(request, Notify)
        self.__requests.add(request)

    def _read_events(self):
        msg = self.__network.recv()
        self.__current_time = msg.now
        return msg.events

    def _send_requests(self):
        msg = Message(self.current_time, list(self.__requests))
        self.__requests.clear()
        self.__network.send(msg)

    def _send_and_recv(self):
        self._send_requests()
        return self._read_events()
