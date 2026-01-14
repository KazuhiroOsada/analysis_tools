import os
import sys
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from analysis.dps import calc_total_energy, calc_SYM_H


savefile = 'symh_summary.pdf'

prefix = '../../../run'
rundirs = [os.path.join(prefix, f'case{irun+1}b128') for irun in range(3)]
runs = []

labels = ['Case 1', 'Case 2', 'Case 3']
colors = ['black', 'red', 'blue']

fig, axes = plt.subplots(2, 1, figsize=(8, 6))
for irun, rundir in enumerate(rundirs):
    run = Run(rundir)
    run.read('coord')
    run.set_trange((0, 1441, 20))
    run.read('moment')

    total_energy = calc_total_energy(run)
    sym_h = calc_SYM_H(run)

    axes[0].text(-0.15, 1.02, 'a', transform=axes[0].transAxes, fontsize=20, fontweight='bold')
    axes[0].plot(run.time/60, total_energy, label=labels[irun], color=colors[irun])
    axes[0].set_ylabel('Total Energy [J]', fontsize=18)
    
    axes[1].text(-0.15, 1.02, 'b', transform=axes[1].transAxes, fontsize=20, fontweight='bold')
    axes[1].plot(run.time/60, sym_h, label=labels[irun], color=colors[irun])
    axes[1].set_ylabel('SYM-H [nT]', fontsize=18)
    axes[1].set_xlabel('Time [min]', fontsize=18)

    for ax in axes:
        ax.legend(fontsize=14)
        ax.grid()
        ax.axvline(20, color='k', linestyle='--')
        ax.axvline(30, color='k', linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim(run.time[0]/60, run.time[-1]/60)

plt.savefig(savefile, bbox_inches='tight')
