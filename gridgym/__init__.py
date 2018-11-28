from gym.envs.registration import register

register(
	id='grid-v0',
	entry_point='gridgym.envs:GridEnv',
)

register(
	id='batsim-v0',
	entry_point='gridgym.envs:BatsimEnv',
)
