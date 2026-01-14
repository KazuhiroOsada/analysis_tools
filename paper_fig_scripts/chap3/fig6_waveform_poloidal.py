import os
import sys
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from analysis.wavelet import bandpass_filter, draw_power_spectrum


savefile = 'wave_poloidal.pdf'

prefix = '../../../run'
rundirs = [os.path.join(prefix, f'case{irun+1}b128') for irun in range(3)]
runs = []

for rundir in rundirs:
    run = Run(rundir)
    run.read_equatorial('bg')
    run.set_trange((0, 1441, 1))
    run.read_equatorial('field')
    run.read_equatorial('moment')
    run.calc_electric_field()
    runs.append(run)

i3, i2 = 5, 4
low_cutoff, high_cutoff = 2.0e-3, 7.0e-3

labels = ['a', 'b', 'c', 'd', 'e']
colors = ['black', 'red', 'blue']
linestyles = ['solid', 'dashed']

fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
colors = ['black', 'red', 'blue']

axes[0].set_title('L = 7.0 R$_\mathrm{E}$ in Case 1, L = 6.0 R$_\mathrm{E}$ in Cases 2 and 3, MLT = 13 h', fontsize=18)
axes[0].text(-0.11, 0.9, labels[0], color='black', fontsize=16, fontweight='bold', transform=axes[0].transAxes)
for irun, run in enumerate(runs):
    axes[0].plot(run.time/60, bandpass_filter(run, run.E[i3, i2, 2, :], low_cutoff, high_cutoff), color = colors[irun], label=f'Case {irun+1}')
axes[0].set_ylabel('E$_\phi$ [mV/m]', fontsize=16)
axes[0].set_ylim(-5e-4, 5e-4)
axes[0].tick_params(axis='y', labelsize=14)
axes[0].ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
axes[0].yaxis.get_offset_text().set_fontsize(14)
axes[0].legend(fontsize=14, loc='upper right', bbox_to_anchor=(1.18, 1.0))

for irun, run in enumerate(runs):
    axes[irun+1].text(-0.11, 0.9, labels[irun+1], color='black', fontsize=16, fontweight='bold', transform=axes[irun+1].transAxes)
    draw_power_spectrum(run, run.E[i3, i2, 2, :], fig=fig, ax=axes[irun+1], xunit='min', yunit='mHz',
                        flog=True, fmin=1e-3, fmax=7e-3, vmax=-4,
                        fontsize=16, label=f'Case{irun+1}, MLT = 13 h', labelsize=16)
    axes[irun+1]._cbar.set_label(f'log$_{{10}}$((E$_\phi$ [mV/m])$^2)$' if irun == 1 else '', fontsize=18)
    axes[irun+1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axes[irun+1].tick_params(axis='y', labelsize=14)

axes[4].text(-0.11, 0.9, labels[4], color='black', fontsize=16, fontweight='bold', transform=axes[4].transAxes)
for irun, run in enumerate(runs):
    axes[4].plot(run.time/60, run.Ppe[i3, i2, :], color = colors[irun])
axes[4].set_xlabel('Time [min]', fontsize=16)
axes[4].set_ylabel('P$_\perp$ [nPa]', fontsize=16)
axes[4].set_xlim(0, 60)
axes[4].set_ylim(-0.05, 0.5)
axes[4].tick_params(axis='x', labelsize=14)
axes[4].tick_params(axis='y', labelsize=14)

plt.savefig(savefile, bbox_inches='tight')
