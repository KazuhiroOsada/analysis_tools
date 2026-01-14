import os
import sys
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from analysis.wavelet import bandpass_filter, draw_power_spectrum


savefile = 'wave_case1.pdf'

rundir = '../../../run/case1b128'

run = Run(rundir)
run.read_equatorial('bg')
run.set_trange((0, 1441, 1))
run.read_equatorial('field')
run.read_equatorial('moment')
run.calc_electric_field()

i3s = [5, 0]
i2 = 4
low_cutoff, high_cutoff = 2.0e-3, 7.0e-3

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
linestyles = ['solid', 'dashed']

fig, axes = plt.subplots(7, 1, figsize=(12, 12), sharex=True)

for i, i3 in enumerate(i3s):
    axes[0].set_title('L = 7.0 R$_\mathrm{E}$, MLT = 13 h and 12 h', fontsize=18)
    axes[0].text(-0.11, 0.9, labels[0], color='black', fontsize=16, fontweight='bold', transform=axes[0].transAxes)
    axes[0].plot(run.time/60, bandpass_filter(run, run.E[i3, i2, 2, :], low_cutoff, high_cutoff), color='black', linestyle=linestyles[i], label=f'MLT = {13 - i} h')
    axes[0].set_ylabel('E$_\phi$ [mV/m]', fontsize=16)
    axes[0].set_ylim(-5e-4, 5e-4)

    axes[1].text(-0.11, 0.9, labels[1], color='black', fontsize=16, fontweight='bold', transform=axes[1].transAxes)

    axes[2].text(-0.11, 0.9, labels[2], color='black', fontsize=16, fontweight='bold', transform=axes[2].transAxes)
    axes[2].plot(run.time/60, bandpass_filter(run, -run.E[i3, i2, 1, :], low_cutoff, high_cutoff), color='black', linestyle=linestyles[i]) # x2 dir is opposite to radial dir
    axes[2].set_ylabel('E$_r$ [mV/m]', fontsize=16)
    axes[2].set_ylim(-5e-4, 5e-4)

    axes[3].text(-0.11, 0.9, labels[3], color='black', fontsize=16, fontweight='bold', transform=axes[3].transAxes)

    axes[4].text(-0.11, 0.9, labels[4], color='black', fontsize=16, fontweight='bold', transform=axes[4].transAxes)
    axes[4].plot(run.time/60, bandpass_filter(run, -run.B[i3, i2, 0, :], low_cutoff, high_cutoff), color='black', linestyle=linestyles[i]) # x1 dir is opposite to parallel dir
    axes[4].set_ylabel('B$_\parallel$ [nT]', fontsize=16)
    
    axes[5].text(-0.11, 0.9, labels[5], color='black', fontsize=16, fontweight='bold', transform=axes[5].transAxes)

    axes[6].text(-0.11, 0.9, labels[6], color='black', fontsize=16, fontweight='bold', transform=axes[6].transAxes)
    axes[6].plot(run.time/60, run.Ppe[i3, i2, :], color='black', linestyle=linestyles[i])
    axes[6].set_ylabel('P$_\perp$ [nPa]', fontsize=16)
    axes[6].set_xlabel('Time [min]', fontsize=18)
    axes[6].set_ylim(-0.05, 0.5)

draw_power_spectrum(run, run.E[i3s[0], i2, 2, :], fig=fig, ax=axes[1], xunit='min', yunit='mHz',
                    flog=True, fmin=1e-3, fmax=7e-3, vmax=-4,
                    fontsize=16, unit=f'E$_\phi$ [mV/m]', label=f'E$_\phi$ MLT = 13 h', labelsize=16)

draw_power_spectrum(run, -run.E[i3s[0], i2, 1, :], fig=fig, ax=axes[3], xunit='min', yunit='mHz',
                    flog=True, fmin=1e-3, fmax=7e-3, vmax=-4,
                    fontsize=16, unit=f'E$_r$ [mV/m]', label=f'E$_r$ MLT = 13 h', labelsize=16)
draw_power_spectrum(run, -run.B[i3s[0], i2, 0, :], fig=fig, ax=axes[5], xunit='min', yunit='mHz',
                    flog=True, fmin=1e-3, fmax=7e-3, vmax=0,
                    fontsize=16, unit=f'B$_\parallel$ [nT]', label=f'B$_\parallel$ MLT = 13 h', labelsize=16)


for ax in [axes[1], axes[3], axes[5]]:
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', labelsize=14)

for ax in [axes[0], axes[2]]:
    ax.tick_params(axis='y', labelsize=14)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))
    ax.yaxis.get_offset_text().set_fontsize(14)

axes[0].legend(fontsize=14, loc='upper right', bbox_to_anchor=(1.22, 1.0))

axes[4].set_ylim(-2e-3, 2e-3)
axes[4].tick_params(axis='y', labelsize=14)
axes[4].ticklabel_format(axis='y', style='sci', scilimits=(-3, -3)) 
axes[4].yaxis.get_offset_text().set_fontsize(14)

axes[6].set_xlim(0, 60)
axes[6].tick_params(axis='x', labelsize=14)
axes[6].tick_params(axis='y', labelsize=14)

plt.savefig(savefile, bbox_inches='tight')
