import os
import sys
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt

from base import Run


savefile = 'L_vs_pressure.pdf'

prefix = '../../../run'
rundirs = [os.path.join(prefix, f'case{irun+1}b128') for irun in range(3)]
runs = []

for i, rundir in enumerate(rundirs):
    run = Run(rundir)
    run.read_equatorial('coord')
    run.set_trange((0, 1441, 20))
    run.read_equatorial('moment')
    run.L = 1 / run.x2**2 / run.Re
    run.Ptot = 1/3 * (run.Ppa + 2*run.Ppe)
    runs.append(run)

ts_min = [30, 120]
its = [round(t_min*60 / (run.delt*run.ifdiag*run.trange[2])) for t_min in ts_min]

i3s = [0, run.N3//4, run.N3//2, 3*run.N3//4]
mlts = [12, 18, 0, 6]

labels = ['a', 'b', 'c', 'd']
colors = ['black', 'red', 'blue', 'gray']
linestyles = ['solid', 'dashed']

fig, axes = plt.subplots(4, 1, figsize=(8, 12))

for irun, run in enumerate(runs):
    for i, i3 in enumerate(i3s):
        for j, it in enumerate(its):
            axes[i].plot(run.L, run.Ptot[i3, :, it], label=f'Case {irun+1}', color=colors[irun], linestyle=linestyles[j])
        axes[i].text(-0.08, 1.0, labels[i], transform=axes[i].transAxes, fontsize=18, fontweight='bold')
        axes[i].text(0.02, 0.9, f'MLT = {mlts[i]} h', transform=axes[i].transAxes, fontsize=18)

        if i == 3:
            axes[i].set_xlabel('L [Re]', fontsize=18)
        axes[i].grid()
        axes[i].tick_params(labelsize=14)

        axes[i].set_xlim(3,8)

for i, ax in enumerate(axes):
    ax.plot(runs[1].L, runs[0].Ptot[i3s[i], :, its[0]], color=colors[-1], linestyle='solid')
    ax.plot(runs[1].L, runs[0].Ptot[i3s[i], :, its[1]], color=colors[-1], linestyle='dashed')

fig.text(0.02, 0.5, 'Pressure [nPa]', va='center', rotation='vertical', fontsize=20)

axes[0].set_ylim(-0.1, 0.7)
axes[1].set_ylim(-0.1, 4)
axes[2].set_ylim(-0.1, 6)
axes[3].set_ylim(-0.1, 5)

from matplotlib.lines import Line2D

line2ds = [
    Line2D([0], [0], color=colors[0], label='Case 1'),
    Line2D([0], [0], color=colors[1], label='Case 2'),
    Line2D([0], [0], color=colors[2],  label='Case 3'),
    Line2D([0], [0], color=colors[3], label='Case 1\n(mapped to $M=2/3$)'),
    Line2D([0], [0], color='none',  label=''),
    Line2D([0], [0], color='k', linestyle='solid',  label='30 min'),
    Line2D([0], [0], color='k', linestyle='dashed', label='120 min'),
]

fig.legend(handles=line2ds,
           loc='upper left',
           bbox_to_anchor=(0.87, 0.9),
           fontsize=13,
           frameon=False)
plt.subplots_adjust(right=0.86, hspace=0.3)

plt.savefig(savefile, bbox_inches='tight')
