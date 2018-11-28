from .batsim import BatsimHandler
from .grid_env import GridEnv


class BatsimEnv(GridEnv):
	def __init__(self):
		super(BatsimEnv, self).__init__()
		self.simulator = BatsimHandler(GridEnv.JOB_SLOTS, GridEnv.TIME_WINDOW, GridEnv.BACKLOG_WIDTH)
