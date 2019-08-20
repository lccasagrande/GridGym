import json as json
import socket
from enum import Enum, auto

import zmq
from procset import ProcSet
from sortedcontainers import SortedList

from .job import Job, JobState
from .resource import *


def message_decoder(msg):
    if "type" in msg:
        if msg["type"] == EventType.JOB_SUBMITTED:
            return JobSubmittedEvent(
                msg["timestamp"],
                msg["data"]["job"]["id"],
                msg["data"]["job"]["profile"],
                msg["data"]["job"]["res"],
                msg["data"]["job"]["walltime"],
            )
        elif msg["type"] == EventType.JOB_COMPLETED:
            return JobCompletedEvent(
                msg["timestamp"],
                msg["data"]["job_id"],
                msg["data"]["job_state"],
                msg["data"]["return_code"],
                msg["data"]["alloc"],
            )
        elif msg["type"] == EventType.JOB_KILLED:
            return JobKilledEvent(msg["timestamp"], msg["data"]["job_ids"])
        elif msg["type"] == EventType.RESOURCE_STATE_CHANGED:
            return ResourceStateChangedEvent(
                msg["timestamp"], msg["data"]["resources"], msg["data"]["state"]
            )
        elif msg["type"] == EventType.REQUESTED_CALL:
            return RequestedCallEvent(msg["timestamp"])
        elif msg["type"] == EventType.SIMULATION_BEGINS:
            return SimulationBeginsEvent(
                msg["timestamp"], msg["data"]["compute_resources"]
            )
        elif msg["type"] == EventType.SIMULATION_ENDS:
            return SimulationEndsEvent(msg["timestamp"])
        elif msg["type"] == EventType.NOTIFY:
            return Notify(msg["timestamp"], msg["data"]["type"])
        else:
            return msg
    elif "now" in msg:
        return Message(msg["now"], msg["events"])
    else:
        return msg


def get_resource_from_json(parent_id, data):
    watt_per_state = data["properties"]["watt_per_state"].split(",")
    s_ps = data["properties"].get("sleep_pstates", None)
    pstate_ids = {i: i for i in range(len(watt_per_state))}

    power_states = []
    if s_ps:
        s_ps = list(map(int, s_ps.split(":")))
        w = float(watt_per_state[pstate_ids.pop(s_ps[0])].split(":")[0])
        power_states.append(PowerState(
            str(s_ps[0]), PowerStateType.sleep, w, w))

        w = float(watt_per_state[pstate_ids.pop(s_ps[1])].split(":")[0])
        power_states.append(
            PowerState(str(s_ps[1]), PowerStateType.switching_off, w, w)
        )

        w = float(watt_per_state[pstate_ids.pop(s_ps[2])].split(":")[0])
        power_states.append(PowerState(
            str(s_ps[2]), PowerStateType.switching_on, w, w))

    for i in pstate_ids.values():
        w = list(map(float, watt_per_state[i].split(":")))
        power_states.append(PowerState(
            str(i), PowerStateType.computation, w[0], w[1]))

    return Resource(
        data["id"],
        parent_id,
        data["name"],
        ResourceState[data["state"]],
        data["properties"]["role"],
        power_states,
    )


def get_free_tcp_address():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", 0))
    host, port = tcp.getsockname()
    tcp.close()
    return "tcp://127.0.0.1:{}".format(port)


class Serializable:
    def __repr__(self):
        return str(self.__dict__)


class WorkloadProfileType(str, Enum):
    delay: str = 'delay'
    parallel: str = 'parallel'
    parallel_homogeneous: str = 'parallel_homogeneous'
    parallel_homogeneous_total: str = 'parallel_homogeneous_total'
    composed: str = 'composed'
    parallel_homogeneous_pfs: str = 'parallel_homogeneous_pfs'
    data_staging: str = 'data_staging'
    smpi: str = 'smpi'


class EventType(str, Enum):
    SIMULATION_BEGINS: str = "SIMULATION_BEGINS"
    SIMULATION_ENDS: str = "SIMULATION_ENDS"
    JOB_SUBMITTED: str = "JOB_SUBMITTED"
    JOB_COMPLETED: str = "JOB_COMPLETED"
    JOB_KILLED: str = "JOB_KILLED"
    RESOURCE_STATE_CHANGED: str = "RESOURCE_STATE_CHANGED"
    REQUESTED_CALL: str = "REQUESTED_CALL"
    NOTIFY: str = "NOTIFY"


class RequestType(str, Enum):
    REJECT_JOB: str = "REJECT_JOB"
    EXECUTE_JOB: str = "EXECUTE_JOB"
    CALL_ME_LATER: str = "CALL_ME_LATER"
    KILL_JOB: str = "KILL_JOB"
    REGISTER_JOB: str = "REGISTER_JOB"
    REGISTER_PROFILE: str = "REGISTER_PROFILE"
    SET_RESOURCE_STATE: str = "SET_RESOURCE_STATE"
    SET_JOB_METADATA: str = "SET_JOB_METADATA"
    CHANGE_JOB_STATE: str = "CHANGE_JOB_STATE"


class NotifyType(str, Enum):
    no_more_static_job_to_submit: str = "no_more_static_job_to_submit"
    no_more_external_event_to_occur: str = "no_more_external_event_to_occur"
    registration_finished: str = "registration_finished"
    continue_registration: str = "continue_registration"


class Message(Serializable):
    def __init__(self, now, events):
        self.now = now
        self.events = events


class Request(Serializable):
    def __init__(self, timestamp, type, data={}):
        assert isinstance(type, RequestType)
        self.timestamp = timestamp
        self.type = type
        self.data = data


class Event():
    def __init__(self, timestamp, type, data={}):
        assert isinstance(type, EventType)
        self.timestamp = timestamp
        self.type = type
        self.data = data


class Notify(Serializable):
    class __data__(Serializable):
        def __init__(self, type):
            self.type = NotifyType[type]

    def __init__(self, timestamp, type):
        self.timestamp = timestamp
        self.type = EventType.NOTIFY
        self.data = self.__data__(type)


class WorkloadProfile(Serializable):
    def __init__(self, type):
        assert isinstance(type, WorkloadProfileType)
        self.type = type


class DelayProfile(WorkloadProfile):
    def __init__(self, delay):
        super().__init__(WorkloadProfileType.delay)
        self.delay = delay


class ParallelProfile(WorkloadProfile):
    def __init__(self, cpu, com):
        super().__init__(WorkloadProfileType.parallel)
        self.cpu = cpu
        self.com = com


class ParallelHomogeneousProfile(WorkloadProfile):
    def __init__(self, cpu, com):
        assert np.isscalar(cpu) and np.isscalar(com)
        super().__init__(WorkloadProfileType.parallel_homogeneous)
        self.cpu = cpu
        self.com = com


class ParallelHomogeneousTotalProfile(WorkloadProfile):
    def __init__(self, cpu, com):
        assert np.isscalar(cpu) and np.isscalar(com)
        super().__init__(WorkloadProfileType.parallel_homogeneous_total)
        self.cpu = cpu
        self.com = com


class JobSubmittedEvent(Event):
    class __data__:
        def __init__(self, id, timestamp, profile, res, walltime):
            self.job_id = id
            self.job = Job(id, res, walltime, profile, timestamp)

    def __init__(self, timestamp, id, profile, res, walltime):
        data = self.__data__(id, timestamp, profile, res, walltime)
        super().__init__(timestamp, EventType.JOB_SUBMITTED, data)


class JobCompletedEvent(Event):
    class __data__:
        def __init__(self, job_id, job_state, return_code, alloc):
            self.job_id = job_id
            self.job_state = JobState[job_state]
            self.return_code = return_code
            self.alloc = list(ProcSet.from_str(alloc))

    def __init__(self, timestamp, job_id, job_state, return_code, alloc):
        data = self.__data__(job_id, job_state, return_code, alloc)
        super().__init__(timestamp, EventType.JOB_COMPLETED, data)


class ResourceStateChangedEvent(Event):
    class __data__:
        def __init__(self, resources, state):
            self.resources = list(ProcSet.from_str(resources))
            self.state = state

    def __init__(self, timestamp, resources, state):
        data = self.__data__(resources, state)
        super().__init__(timestamp, EventType.RESOURCE_STATE_CHANGED, data)


class JobKilledEvent(Event):
    class __data__:
        def __init__(self, job_ids):
            assert isinstance(job_ids, list)
            self.job_ids = job_ids

    def __init__(self, timestamp, job_ids):
        data = self.__data__(job_ids)
        super().__init__(timestamp, EventType.JOB_KILLED, data)


class RequestedCallEvent(Event):
    def __init__(self, timestamp):
        super().__init__(timestamp, EventType.REQUESTED_CALL)


class SimulationBeginsEvent(Event):
    class __data__:
        def __init__(self, compute_resources):
            node_id, nodes = 0, []
            while compute_resources:
                zone_properties = compute_resources[0].get(
                    "zone_properties", {"cores_per_node": "1"}
                )
                cores_per_node = int(zone_properties["cores_per_node"])
                resources = [
                    get_resource_from_json(node_id, data)
                    for data in compute_resources[:cores_per_node]
                ]
                del compute_resources[:cores_per_node]
                nodes.append(Node(node_id, resources))
                node_id += 1

            self.platform = Platform(nodes)

    def __init__(self, timestamp, compute_resources):
        data = self.__data__(compute_resources)
        super().__init__(timestamp, EventType.SIMULATION_BEGINS, data)


class SimulationEndsEvent(Event):
    def __init__(self, timestamp):
        super().__init__(timestamp, EventType.SIMULATION_ENDS)


class RejectJobRequest(Request):
    class __data__(Serializable):
        def __init__(self, job_id):
            self.job_id = job_id

    def __init__(self, timestamp, job_id):
        data = self.__data__(job_id)
        super().__init__(timestamp, RequestType.REJECT_JOB, data)


class ExecuteJobRequest(Request):
    class __data__(Serializable):
        def __init__(self, job_id, alloc):
            assert isinstance(alloc, list)
            self.job_id = job_id
            self.alloc = str(ProcSet(*alloc))

    def __init__(self, timestamp, job_id, alloc):
        data = self.__data__(job_id, alloc)
        super().__init__(timestamp, RequestType.EXECUTE_JOB, data)


class CallMeLaterRequest(Request):
    class __data__(Serializable):
        def __init__(self, timestamp):
            self.timestamp = timestamp

    def __init__(self, timestamp, when):
        data = self.__data__(when)
        super().__init__(timestamp, RequestType.CALL_ME_LATER, data)


class KillJobRequest(Request):
    class __data__(Serializable):
        def __init__(self, job_ids):
            self.job_ids = job_ids

    def __init__(self, timestamp, job_ids):
        data = __data__(job_ids)
        super().__init__(timestamp, RequestType.KILL_JOB, data)


class RegisterJobRequest(Request):
    class __job__(Serializable):
        def __init__(self, id, profile, res, walltime):
            self.profile = profile
            self.res = res
            self.id = id
            self.walltime = walltime

    class __data__(Serializable):
        def __init__(self, id, profile, res, walltime):
            self.job_id = id
            self.job = __job__(id, profile, res, walltime)

    def __init__(self, timestamp, id, profile, res, walltime):
        data = __data__(id, profile, res, walltime)
        super().__init__(timestamp, RequestType.REGISTER_JOB, data)


class RegisterProfileRequest(Request):
    class __data__(Serializable):
        def __init__(self, workload_name, profile_name, profile):
            assert isinstance(profile, WorkloadProfile)
            self.workload_name = workload_name
            self.profile_name = profile_name
            self.profile = profile

    def __init__(self, timestamp, workload_name, profile_name, profile):
        data = self.__data__(workload_name, profile_name, profile)
        super().__init__(timestamp, RequestType.REGISTER_PROFILE, data)


class SetResourceStateRequest(Request):
    class __data__(Serializable):
        def __init__(self, resources, state):
            assert isinstance(resources, list)
            self.resources = str(ProcSet(*resources))
            self.state = state

    def __init__(self, timestamp, resources, state):
        data = self.__data__(resources, state)
        super().__init__(timestamp, RequestType.SET_RESOURCE_STATE, data)


class ChangeJobStateRequest(Request):
    class __data__(Serializable):
        def __init__(self, job_id, job_state, kill_reason):
            self.job_id = job_id
            self.job_state = job_state
            self.kill_reason = kill_reason

    def __init__(self, timestamp, job_id, job_state, kill_reason):
        data = self.__data__(job_id, job_state, kill_reason)
        super().__init__(timestamp, RequestType.CHANGE_JOB_STATE, data)


class MessageEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Serializable):
            return repr(o) if isinstance(o, Enum) else o.__dict__
        return json.JSONEncoder.default(self, o)


class NetworkHandler:
    def __init__(self, address, type=zmq.REP):
        self.address = address
        self.context = zmq.Context()
        self.socket = None
        self.type = type

    def send(self, msg):
        assert self.socket, "Connection not open"
        self.socket.send_json(msg, cls=MessageEncoder)

    def recv(self):
        assert self.socket, "Connection not open"
        return self.socket.recv_json(object_hook=message_decoder)

    def _send_and_recv(self, msg):
        self.send(msg)
        return self.recv()

    def bind(self):
        assert not self.socket, "Connection already open"
        self.socket = self.context.socket(self.type)
        self.socket.bind(self.address)

    def connect(self):
        assert not self.socket, "Connection already open"
        self.socket = self.context.socket(self.type)
        self.socket.connect(self.address)

    def close(self):
        if self.socket:
            self.socket.close()
            self.socket = None


class ProtocolHandler:
    WORKLOAD_JOB_SEPARATOR = "!"
    ATTEMPT_JOB_SEPARATOR = "#"
    WORKLOAD_JOB_SEPARATOR_REPLACEMENT = "%"

    def __init__(self, address=None):
        if address is None:
            address = get_free_tcp_address()
        self._network = NetworkHandler(address)
        self._requests = SortedList([], key=lambda r: r.timestamp)
        self._current_time = 0.0
        self.__callbacks = {k: [] for k in EventType}

    @property
    def address(self):
        return self._network.address

    @property
    def current_time(self):
        return self._current_time  # float("%4.f" % self._current_time)

    def ack(self):
        self._network.send(Message(self.current_time, []))

    def set_callback(self, event_type, call):
        self.__callbacks[event_type].append(call)

    def proceed_simulation(self):
        self._handle_events(self._send_and_recv())

    def start(self):
        self._network.bind()
        self._handle_events(self._read_events())

    def close(self):
        self._requests.clear()
        self._current_time = 0.0
        self._network.close()

    def execute_job(self, job_id, alloc):
        request = ExecuteJobRequest(self.current_time, job_id, alloc)
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

    def set_resources_state(self, resources, state):
        request = SetResourceStateRequest(self.current_time, resources, state)
        self._append_request(request)

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
        self._requests.add(request)

    def _handle_events(self, events):
        for e in events:
            for callback in self.__callbacks[e.type]:
                callback(e.data)

    def _read_events(self):
        msg = self._network.recv()
        self._current_time = msg.now
        return msg.events

    def _send_requests(self):
        msg = Message(self.current_time, list(self._requests))
        self._requests.clear()
        self._network.send(msg)

    def _send_and_recv(self):
        self._send_requests()
        return self._read_events()

# msg = '{"now":0.000000,"events":[{"timestamp":0.000000,"type":"SIMULATION_BEGINS","data":{"nb_resources":168,"nb_compute_resources":168,"nb_storage_resources":0,"allow_compute_sharing":false,"allow_storage_sharing":true,"config":{"redis-enabled":false,"redis-hostname":"127.0.0.1","redis-port":6379,"redis-prefix":"default","profiles-forwarded-on-submission":false,"dynamic-jobs-enabled":false,"dynamic-jobs-acknowledged":false,"forward-unknown-events":false},"compute_resources":[{"id":0,"name":"0","state":"idle","properties":{"role":"","watt_off":"9","watt_per_state":"9:9,95.0:190.0,125.0:125.0,101.0:101.0","sleep_pstates":"0:3:2"},"zone_properties":{"cores_per_node":"1"}},{"id":1,"name":"1","state":"idle","properties":{"role":"","watt_off":"9","watt_per_state":"9:9,95.0:190.0,125.0:125.0,101.0:101.0","sleep_pstates":"0:3:2"},"zone_properties":{"cores_per_node":"1"}},{"id":2,"name":"10","state":"idle","properties":{"role":"","watt_off":"9","watt_per_state":"9:9,95.0:190.0,125.0:125.0,101.0:101.0","sleep_pstates":"0:3:2"},"zone_properties":{"cores_per_node":"1"}}]}}]}'
