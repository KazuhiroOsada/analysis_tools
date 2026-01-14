import os
import sys
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt


savefile = 'wave_characteristics.pdf'

prefix = '../../analysis/processed'
cwt_analysis_files = [os.path.join(prefix, 'analysis', f'case{irun+1}b128.npz') for irun in range(3)]
gamma_raw_files = [os.path.join(prefix, 'growthrate', f'case{irun+1}b128_raw.npz') for irun in range(3)]
gamma_ma_files = [os.path.join(prefix, 'growthrate', f'case{irun+1}b128_ma.npz') for irun in range(3)]

i3, i2 = 5, 4

colors = ['black', 'red', 'blue']

fig, axes = plt.subplots(5, 1, figsize=(12,12), sharex=True)

labels = ['a', 'b', 'c', 'd', 'e']
for i, label in enumerate(labels):
    axes[i].text(-0.13, 1.0, label, color='black', fontsize=16, fontweight='bold', transform=axes[i].transAxes)

axes[0].set_title('L = 7.0 R$_\mathrm{E}$ in Case 1, L = 6.0 R$_\mathrm{E}$ in Cases 2 and 3, MLT = 13 h', fontsize=18)
for irun, (cwt_file, gamma_raw_file, gamma_ma_file) in enumerate(zip(cwt_analysis_files, gamma_raw_files, gamma_ma_files)):
    data = np.load(cwt_file)
    time_min = data['time'] / 60
    max_power_freq = data['max_power_freq']
    max_power = data['max_power']
    mnumbers = data['mnumbers']

    mask = max_power_freq < 2.0e-3
    max_power_freq[mask] = np.nan
    max_power[mask] = np.nan
    mnumbers[mask] = np.nan

    gamma_raw_data = np.load(gamma_raw_file)
    time_g = gamma_raw_data['time'] / 60
    gamma_raw = gamma_raw_data['gamma']
    w_res = gamma_raw_data['w_res']

    gamma_ma_data = np.load(gamma_ma_file)
    time_g = gamma_ma_data['time'] / 60
    gamma_ma = gamma_ma_data['gamma_ma']

    axes[0].plot(time_min, np.log10(max_power[i3, i2, :]), color=colors[irun], label=f'Case {irun+1}')
    axes[0].set_ylabel('log$_{10}$(Power\n[(mV/m)$^2$])', fontsize=16)
    axes[0].set_ylim(-6, -5)
    axes[0].tick_params(axis='y', labelsize=14)
    axes[0].legend(fontsize=14)

    axes[1].plot(time_min, max_power_freq[i3, i2, :]*1e3, color=colors[irun])
    axes[1].set_ylabel('Frequency\n[mHz]', fontsize=16)
    axes[1].set_ylim(2, 4)
    axes[1].tick_params(axis='y', labelsize=14)

    axes[2].plot(time_min, mnumbers[i3, i2, :], color=colors[irun])
    axes[2].set_ylabel('m number', fontsize=16)
    axes[2].set_ylim(-20, -5)
    axes[2].tick_params(axis='y', labelsize=14)

    axes[3].plot(time_g, gamma_ma[i3, i2, :], '.-', color=colors[irun])
    axes[3].plot(time_g, gamma_raw[i3, i2, :],'.-', color=colors[irun], alpha=0.3)
    axes[3].set_ylabel('Growth rate\n[/s]', fontsize=16)
    axes[3].set_ylim(0, 4e-3)
    axes[3].tick_params(axis='y', labelsize=14)
    axes[3].ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
    axes[3].yaxis.get_offset_text().set_fontsize(14)

    axes[4].plot(time_g, w_res[i3, i2, :], '.-', color=colors[irun])
    axes[4].set_ylabel('Resonant energy\n[keV]', fontsize=16)
    axes[4].set_xlabel('Time [min]', fontsize=18)
    axes[4].set_ylim(0, 125)
    axes[4].tick_params(axis='y', labelsize=14)
    axes[4].set_xlim(15, 37)
    axes[4].tick_params(axis='x', labelsize=14)

plt.savefig(savefile, bbox_inches='tight')
