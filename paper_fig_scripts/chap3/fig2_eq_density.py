import os
import sys
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from draw import draw_equatorial
from analysis.calc_potential import calc_potential


savefile = 'density_snapshot.pdf'

prefix = '../../../run'
rundirs = [os.path.join(prefix, f'case{irun+1}b128') for irun in range(3)]
runs = []

for irun, rundir in enumerate(rundirs):
    run = Run(rundir)
    run.read_equatorial('bg')
    run.read_equatorial('coord')
    run.set_trange((0, 1441, 120))
    run.read_equatorial('moment')
    run.read_equatorial('field')
    run.calc_electric_field()
    runs.append(run)
    print(f'data read completed for case {irun+1}')

ts_min = [20, 30, 120]
its = [round(t_min*60 / (run.delt*run.ifdiag*run.trange[2])) for t_min in ts_min]

fig, axes = plt.subplots(3, 3, figsize=(12,12))

labels = [['a', 'b', 'c'],
          ['d', 'e', 'f'],
          ['g', 'h', 'i']]
tick = np.linspace(-8.0, 8.0, 9)

for irun, run in enumerate(runs):
    for i, it in enumerate(its):
        pot = calc_potential(run, run.E[..., it])
        pcm = draw_equatorial(run, run.Rho[..., it], fig, axes[i, irun],
                              vmin=10, vmax=1000, log=True,
                              title=f'Case {irun+1}' if i==0 else None, width=8.0,
                              ylabel=(irun==0), xlabel=(i==2), colorbar=False, return_pcm=True)
        ctr = axes[i,irun].contour(run.Xi, run.Yi, pot, colors='white', linewidths=1.0, levels=np.arange(-40, 41, 10))
        axes[i,irun].contour(run.Xi, run.Yi, pot, colors='white', linewidths=0.3, levels=np.arange(-40, 41, 2))
        axes[i,irun].clabel(ctr, fmt='%d', colors='white', fontsize=8)

        axes[i,irun].text(-0.15, 1.05, labels[i][irun],
                          transform=axes[i,irun].transAxes, fontsize=20, fontweight='bold')
        axes[i,irun].set_xticks(tick)
        axes[i,irun].set_yticks(tick)

        # wave analysis point
        i2 = 4
        if irun == 0:
            i3s = np.array([0, 5])
        else:
            i3s = 5
        axes[i,irun].scatter(run.Xh[i3s, i2], run.Yh[i3s, i2], c='purple', marker='x')

        if irun == 0:
            axes[i,irun].text(-0.35, 0.5, f't = {ts_min[i]} min',
                              transform=axes[i,irun].transAxes, fontsize=16, rotation=90, va='center')

        if irun == 2:
            pos = axes[i,irun].get_position()
            cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
            cbar = fig.colorbar(pcm, cax=cbar_ax)
            cbar_ax.tick_params(labelsize=14)
            cbar.set_label('Density [/cc]', fontsize=16)

plt.savefig(savefile, bbox_inches='tight')
