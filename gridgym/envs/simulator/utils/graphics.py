import time
import os

import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import numpy as np

from evalys.jobset import *
from evalys.mstates import *
from evalys.pstates import *
from evalys.visu.legacy import *


def plot_simulation_graphics(output_dir, show=True):
	jobs = JobSet.from_csv(os.path.join(output_dir, '_jobs.csv'))
	mstates = MachineStatesChanges(os.path.join(output_dir, '_machine_states.csv')).df
	pstates = PowerStatesChanges(os.path.join(output_dir, '_pstate_changes.csv') )
	energy = pd.read_csv(os.path.join(output_dir, '_consumed_energy.csv') )
	df = energy.drop_duplicates(subset='time')
	df = df.drop(['event_type', 'wattmin', 'epower'], axis=1)
	diff = df.diff(1)
	diff.rename(columns={'time': 'time_diff', 'energy': 'energy_diff'}, inplace=True)
	power = pd.concat([df, diff], axis=1)
	power['power'] = power['energy_diff'] / power['time_diff']

	fig, ax_list = plt.subplots(6, sharex=True, sharey=False, figsize=(32,22))
	fig.subplots_adjust(bottom=0.05, right=0.95, top=0.95, left=0.05)
	plot_load(jobs.utilisation, jobs.MaxProcs, legend_label='Resource Load', ax=ax_list[0])
	plot_load(jobs.queue, jobs.MaxProcs, legend_label='Queue Load', ax=ax_list[1])
	plot_job_details(jobs.df, jobs.MaxProcs, ax=ax_list[2])
	plot_gantt(jobs, ax=ax_list[3], title="Gantt")
#
	plot_gantt_pstates(
		jobs,
		pstates,
		ax_list[4],
		title="Gantt chart (Pstates)",
		labels=False,
		off_pstates=set([0]),
		son_pstates=set([2]),
		soff_pstates=set([3]))
#
	plot_mstates(mstates, ax_list[5], title="Resources state")
	ax_list[5].legend(loc='center left', bbox_to_anchor=(1, 0.5))
	#ax_list[6].plot(power['time'], power['power'], drawstyle='steps-pre')
	#ax_list[6].set_title('Power (W)')
	#ax_list[7].plot(energy['time'], energy['energy'])
	#ax_list[7].set_title('Energy (J)')
	if show:
		#plt.xticks(np.arange(power['time'].min(), power['time'].max() + 1, 1.0))
		plt.show()
	return fig