from .batsim import BatsimHandler
from .grid_env import GridEnv


class BatsimEnv(GridEnv):
	def __init__(self):
		super(BatsimEnv, self).__init__()
		self.simulator = BatsimHandler(GridEnv.NB_JOB_SLOTS, GridEnv.TIME_SLICE, GridEnv.BACKLOG_WIDTH)
