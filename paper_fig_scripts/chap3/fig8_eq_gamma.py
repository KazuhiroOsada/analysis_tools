import os
import sys
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from draw import draw_equatorial


savefile = 'gamma_equator.pdf'

run_prefix = '../../../run'
rundirs = [os.path.join(run_prefix, f'case{irun+1}b128') for irun in range(3)]
gamma_prefix = '../../analysis/processed/growthrate'
gamma_files = [os.path.join(gamma_prefix, f'case{irun+1}b128_ma.npz') for irun in range(3)]
runs = []

t_min = 20

for irun, rundir in enumerate(rundirs):
    run = Run(rundir)
    run.read_equatorial('coord')

    data = np.load(gamma_files[irun])
    time = data['time']
    it = np.argmin(np.abs(time - t_min*60))
    run.gamma = data['gamma_ma'][..., it]

    runs.append(run)

labels = ['a', 'b', 'c']

fig, axes = plt.subplots(1, 3, figsize=(15,5))
axes[1].text(0.5, 1.15, f't = {t_min} min', fontsize=24, transform=axes[1].transAxes, ha='center')
for irun, run in enumerate(runs):
    axes[irun].text(-0.16, 1.05, labels[irun], fontsize=20, fontweight='bold', transform=axes[irun].transAxes)
    pcm = draw_equatorial(run, run.gamma, fig=fig, ax=axes[irun],
                          vmin=-3e-3, vmax=3e-3, gridline=True, gridline_alpha=0.1, cmap='coolwarm', ylabel=irun == 0,
                          title=f'Case {irun+1}', colorbar=False, return_pcm=True, width=8)
    
    # wave analysis points
    i2 = 4
    if irun == 0:
        i3s = np.array([0, 5])
    else:
        i3s = 5
    axes[irun].scatter(run.Xh[i3s, i2], run.Yh[i3s, i2], c='purple', marker='x')

pos = axes[2].get_position()
cbar_ax = fig.add_axes([pos.x0 + pos.width + 0.02, pos.y0, 0.02, pos.height])
cbar = fig.colorbar(pcm, cax=cbar_ax)
cbar.set_label('Growth rate $\gamma$ [/s]', fontsize=16)
cbar_ax.tick_params(labelsize=14)
cbar_ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
cbar_ax.yaxis.get_offset_text().set_fontsize(14)

plt.savefig(savefile, bbox_inches='tight')
