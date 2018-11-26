from gym.envs.registration import register

register(
	id='grid-v0',
	entry_point='GridEnv.envs:GridEnv',
)

register(
	id='batsim-v0',
	entry_point='GridEnv.envs:BatsimEnv',
)
