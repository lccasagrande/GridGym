from gym.envs.registration import register

register(
	id='OffReservation-v0',
	entry_point='gridgym.envs:OffReservationEnv',
)

register(
	id='Scheduling-v0',
	entry_point='gridgym.envs:SchedulingEnv',
)