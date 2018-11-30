from gym.envs.registration import register

register(
	id='grid-v0',
	entry_point='GridGym.gridgym.envs:GridEnv',
)

register(
	id='batsim-v0',
	entry_point='GridGym.gridgym.envs:BatsimEnv',
)
