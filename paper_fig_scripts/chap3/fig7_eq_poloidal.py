import os
import sys
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from draw import draw_equatorial


savefile = 'wave_equator.pdf'

run_prefix = '../../../run'
rundirs = [os.path.join(run_prefix, f'case{irun+1}b128') for irun in range(3)]
analysis_prefix = '../../analysis/processed/analysis'
analysis_files = [os.path.join(analysis_prefix, f'case{irun+1}b128.npz') for irun in range(3)]
runs = []

for irun, rundir in enumerate(rundirs):
    run = Run(rundir)
    run.read_equatorial('coord')
    run.set_trange((240, 241, 1))
    run.read_equatorial('moment')

    data = np.load(analysis_files[irun])
    time_slice = slice(run.trange[0], run.trange[1], run.trange[2])
    run.max_power_freq = data['max_power_freq'][..., time_slice]
    run.max_power = data['max_power'][..., time_slice]
    run.mnumbers = data['mnumbers'][..., time_slice]

    mask = run.max_power_freq < 2e-3
    run.max_power_freq[mask] = np.nan
    run.max_power[mask] = np.nan
    run.mnumbers[mask] = np.nan

    runs.append(run)

labels = [['a', 'b', 'c', 'd'],
          ['e', 'f', 'g', 'h'],
          ['i', 'j', 'k', 'l']]
pcms = np.zeros(4, dtype=object)

fig, axes = plt.subplots(3, 4, figsize=(16,13.5))
axes[0, 1].text(1.0, 1.15, 't = 20 min', fontsize=24, transform=axes[0,1].transAxes, ha='center')
for irun, run in enumerate(runs):
    axes[irun, 0].text(-0.16, 0.95, labels[irun][0], fontsize=20, fontweight='bold', transform=axes[irun,0].transAxes)
    pcms[0] = draw_equatorial(run, run.Ppe[..., 0], fig=fig, ax=axes[irun,0],
                              vmin=1e-4, vmax=10, log=True, xlabel=irun==2, ylabel=True,
                              title='P$_\perp$ [nPa]' if irun==0 else None, colorbar=False, return_pcm=True, width=8)
    axes[irun, 1].text(-0.16, 0.95, labels[irun][1], fontsize=20, fontweight='bold', transform=axes[irun,1].transAxes)
    pcms[1] = draw_equatorial(run, np.log10(run.max_power[..., 0]), fig=fig, ax=axes[irun,1],
                              vmin=-6.5, vmax=-4.5, gridline=True, gridline_alpha=0.1,  xlabel=irun==2, ylabel=False,
                              title='log$_{10}$(Power [(mV/m)$^2$])' if irun==0 else None, colorbar=False, return_pcm=True, width=8)
    axes[irun, 2].text(-0.16, 0.95, labels[irun][2], fontsize=20, fontweight='bold', transform=axes[irun,2].transAxes)
    pcms[2] = draw_equatorial(run, run.max_power_freq[..., 0]*1e3, fig=fig, ax=axes[irun,2],
                              vmin=2, vmax=8, gridline=True, gridline_alpha=0.1,  xlabel=irun==2, ylabel=False,
                              title='Frequency [mHz]' if irun==0 else None, colorbar=False, return_pcm=True, width=8)
    axes[irun, 3].text(-0.16, 0.95, labels[irun][3], fontsize=20, fontweight='bold', transform=axes[irun,3].transAxes)
    pcms[3] = draw_equatorial(run, run.mnumbers[..., 0], fig=fig, ax=axes[irun,3],
                              vmin=-20, vmax=20, gridline=True, gridline_alpha=0.1, cmap='coolwarm',  xlabel=irun==2, ylabel=False,
                              title='m number' if irun==0 else None, colorbar=False, return_pcm=True, width=8)
    axes[irun, 0].text(-0.35, 0.5, f'Case {irun+1}', color='black', fontsize=20, 
                        transform=axes[irun,0].transAxes, rotation=90, va='center')

    # wave analysis points
    for i in range(4):
        i2 = 4
        if irun == 0:
            i3s = np.array([0, 5])
        else:
            i3s = 5
        axes[irun, i].scatter(run.Xh[i3s, i2], run.Yh[i3s, i2], c='purple', marker='x')

cbar_labels = ['P$_\perp$ [nPa]', 'log$_{10}$(Power [(mV/m)$^2$])', 'Frequency [mHz]', 'm number']
for i in range(4):
    ax = axes[2, i]
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0, pos.y0 - 0.045 - 0.018, pos.width, 0.018])
    fig.colorbar(pcms[i], cax=cax, orientation='horizontal')
    cax.set_xlabel(cbar_labels[i], fontsize=18)
    cax.tick_params(labelsize=14)

plt.savefig(savefile, bbox_inches='tight')
