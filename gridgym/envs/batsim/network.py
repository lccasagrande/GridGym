import json
import socket
import zmq
from .scheduler import Job


class BatsimProtocolHandler:
    WORKLOAD_JOB_SEPARATOR = "!"
    ATTEMPT_JOB_SEPARATOR = "#"
    WORKLOAD_JOB_SEPARATOR_REPLACEMENT = "%"

    def get_free_tcp_address(self):
        tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp.bind(('', 0))
        host, port = tcp.getsockname()
        tcp.close()
        return 'tcp://{host}:{port}'.format(**locals())

    def __init__(self, socket_endpoint=None):
        if socket_endpoint == None:
            self.socket_endpoint = self.get_free_tcp_address()
        else:
            self.socket_endpoint = socket_endpoint
            
        self._network = NetworkHandler(self.socket_endpoint)
        self.reset()

    def reset(self):
        self.events = []
        self.current_time = 0.0
        self._ack = False


    def read_events(self, blocking):
        def get_msg():
            msg = None
            while msg is None:
                msg = self._network.recv(blocking=blocking)
                if msg is None:
                    raise ValueError(
                        "Batsim is not responding (maybe deadlocked)")
            return BatsimMessage.from_json(msg)

        msg = get_msg()

        self.update_time(msg.now)

        return msg.events

    def send_events(self):
        if not self.has_events():
            return

        msg = self._flush()
        assert msg is not None, "Cannot send a message if no event ocurred."
        self._network.send(msg)

    def start(self):
        self.reset()
        self._network.bind()

    def close(self):
        self._network.close()

    def _flush(self):
        self._ack = False

        if len(self.events) > 0:
            self.events = sorted(
                self.events, key=lambda event: event['timestamp'])

        msg = {
            "now": self.current_time,
            "events": self.events
        }

        self.events = []

        return msg

    def has_events(self):
        return len(self.events) > 0 or self._ack

    @property
    def current_time(self):
        return self._current_time

    @current_time.setter
    def current_time(self, value):
        self._current_time = float("%4.f" % value)

    def update_time(self, time):
        self.current_time = time

    def consume_time(self, time):
        self.current_time += time
        return self.current_time

    def set_alarm(self, time):
        self.events.append(
            {"timestamp": self.current_time,
             "type": "CALL_ME_LATER",
             "data": {"timestamp": time}})

    def notify_submission_finished(self):
        self.events.append({
            "timestamp": self.current_time,
            "type": "NOTIFY",
            "data": {
                    "type": "submission_finished",
            }
        })

    def notify_submission_continue(self):
        self.events.append({
            "timestamp": self.current_time,
            "type": "NOTIFY",
            "data": {
                    "type": "continue_submission",
            }
        })

    def send_message_to_job(self, job, message):
        self.events.append({
            "timestamp": self.current_time,
            "type": "TO_JOB_MSG",
            "data": {
                    "job_id": job.id,
                    "msg": message,
            }
        })

    def start_job(self, job_id, res):
        """ args:res: is list of int (resources ids) """
        self.events.append({
            "timestamp": self.current_time,
            "type": "EXECUTE_JOB",
            "data": {
                    "job_id": job_id,
                    "alloc": " ".join(str(r) for r in res)
            }
        })

    def execute_jobs(self, jobs, io_jobs=None):
        """ args:jobs: list of jobs to execute (job.allocation MUST be set) """

        for job in jobs:
            assert job.allocation is not None

            message = {
                "timestamp": self.current_time,
                "type": "EXECUTE_JOB",
                "data": {
                        "job_id": job.id,
                        "alloc": str(job.allocation)
                }
            }
            if io_jobs is not None and job.id in io_jobs:
                message["data"]["additional_io_job"] = io_jobs[job.id]

            self.events.append(message)

    def reject_job(self, job_id):
        """Reject the given jobs."""
        self.events.append({
            "timestamp": self.current_time,
            "type": "REJECT_JOB",
            "data": {
                    "job_id": job_id,
            }
        })

    def change_state(self, job, state):
        """Change the state of a job."""
        self.events.append({
            "timestamp": self.current_time,
            "type": "CHANGE_state",
            "data": {
                    "job_id": job.id,
                    "state": state.name,
            }
        })

    def kill_jobs(self, jobs):
        """Kill the given jobs."""
        assert len(jobs) > 0, "The list of jobs to kill is empty"
        for job in jobs:
            job.state = Job.State.IN_KILLING
        self.events.append({
            "timestamp": self.current_time,
            "type": "KILL_JOB",
            "data": {
                    "job_ids": [job.id for job in jobs],
            }
        })

    def submit_profiles(self, workload_name, profiles):
        for profile_name, profile in profiles.items():
            msg = {
                "timestamp": self.current_time,
                "type": "SUBMIT_PROFILE",
                "data": {
                    "workload_name": workload_name,
                    "profile_name": profile_name,
                    "profile": profile,
                }
            }
            self.events.append(msg)

    def submit_job(
            self,
            id,
            res,
            walltime,
            profile_name,
            subtime=None,
            profile=None):

        job_dict = {
            "profile": profile_name,
            "id": id,
            "res": res,
            "walltime": walltime,
            "subtime": self.current_time if subtime is None else subtime,
        }
        msg = {
            "timestamp": self.current_time,
            "type": "SUBMIT_JOB",
            "data": {
                "job_id": id,
                "job": job_dict,
            }
        }
        if profile is not None:
            assert isinstance(profile, dict)
            msg["data"]["profile"] = profile

        self.events.append(msg)

        return id

    def set_resource_pstate(self, resources, state):
        self.events.append({
            "timestamp": self.current_time,
            "type": "SET_RESOURCE_STATE",
            "data": {
                "resources": " ".join([str(r) for r in resources]),
                "state": str(state.value)
            }
        })

    def request_consumed_energy(self):  # TODO CHANGE NAME
        self.events.append(
            {
                "timestamp": self.current_time,
                "type": "QUERY",
                "data": {
                    "requests": {"consumed_energy": {}}
                }
            }
        )

    def request_air_temperature_all(self):
        self.events.append(
            {
                "timestamp": self.current_time,
                "type": "QUERY",
                "data": {
                    "requests": {"air_temperature_all": {}}
                }
            }
        )

    def request_processor_temperature_all(self):
        self.events.append(
            {
                "timestamp": self.current_time,
                "type": "QUERY",
                "data": {
                    "requests": {"processor_temperature_all": {}}
                }
            }
        )

    def notify_resources_added(self, resources):
        self.events.append(
            {
                "timestamp": self.current_time,
                "type": "RESOURCES_ADDED",
                "data": {
                    "resources": resources
                }
            }
        )

    def notify_resources_removed(self, resources):
        self.events.append(
            {
                "timestamp": self.current_time,
                "type": "RESOURCES_REMOVED",
                "data": {
                    "resources": resources
                }
            }
        )

    def set_job_metadata(self, job_id, metadata):
        # Consume some time to be sure that the job was created before the
        # metadata is set

        if self.events == None:
            self.events = []
        self.events.append(
            {
                "timestamp": self.current_time,
                "type": "SET_JOB_METADATA",
                "data": {
                    "job_id": str(job_id),
                    "metadata": str(metadata)
                }
            }
        )

    def resubmit_job(self, job, delay=1):
        job_id = job.id
        if job.id.find(BatsimProtocolHandler.ATTEMPT_JOB_SEPARATOR) == -1:
            job_id += BatsimProtocolHandler.ATTEMPT_JOB_SEPARATOR + "1"
        else:
            job_id = job_id[:-1] + str(int(job_id[-1]) + 1)

        self.reject_job(job.id)

        self.consume_time(delay)

        self.submit_job(
            job_id,
            job.requested_resources,
            job.requested_time,
            job.profile,
            profile=job.profile_dict,
            subtime=job.submit_time)

    def acknowledge(self):
        self._ack = True


class NetworkHandler:

    def __init__(
            self,
            socket_endpoint,
            verbose=0,
            timeout=10000,
            type=zmq.REP):
        self.socket_endpoint = socket_endpoint
        self.verbose = verbose
        self.timeout = timeout
        self.context = zmq.Context()
        self.connection = None
        self.type = type

    def send(self, msg):
        self.send_string(json.dumps(msg))

    def send_string(self, msg):
        assert self.connection, "Connection not open"
        self.connection.send_string(msg)

    def recv(self, blocking=False):
        msg = self.recv_string(blocking=blocking)
        if msg is not None:
            msg = json.loads(msg)
        return msg

    def recv_string(self, blocking=False):
        assert self.connection, "Connection not open"
        if blocking or self.timeout is None or self.timeout <= 0:
            self.connection.RCVTIMEO = -1
        else:
            self.connection.RCVTIMEO = self.timeout
        try:
            msg = self.connection.recv_string()
        except zmq.error.Again:
            return None

        return msg

    def bind(self):
        assert not self.connection, "Connection already open"
        self.connection = self.context.socket(self.type)
        self.connection.bind(self.socket_endpoint)

    def connect(self):
        assert not self.connection, "Connection already open"
        self.connection = self.context.socket(self.type)
        self.connection.connect(self.socket_endpoint)

    def subscribe(self, pattern=b''):
        self.type = zmq.SUB
        self.connect()
        self.connection.setsockopt(zmq.SUBSCRIBE, pattern)

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None


class BatsimEvent:
    def __init__(self, timestamp, type, data):
        self.timestamp = float("{:.4f}".format(float(timestamp)))
        self.type = type
        self.data = data


class BatsimMessage:
    def __init__(self, now, events):
        self.now = float("%4.f" % now)
        self.events = [BatsimEvent(
            event['timestamp'], event['type'], event['data']) for event in events]

    @staticmethod
    def from_json(data):
        return BatsimMessage(data['now'], data['events'])
