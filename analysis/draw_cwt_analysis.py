import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from draw import draw_equatorial


def read_cwt_analysis_data(filename):
    data = np.load(filename)
    max_power_freq = data['max_power_freq']
    max_power      = data['max_power']
    mnumbers       = data['mnumbers']
    return max_power_freq, max_power, mnumbers


def compare_equatorial():
    filenames = ['cwt_analysis_case1_Pc5_Ephi.npz',
                 'cwt_analysis_case2_Pc5_Ephi.npz',
                 'cwt_analysis_case3_Pc5_Ephi.npz']
    rundirs = ['case1b256', 'case2b256new', 'case3b256']

    runs = []
    for filename, rundir in zip(filenames, rundirs):
        max_power_freq, max_power, mnumbers = read_cwt_analysis_data(filename)
        run = Run(f'../../run/{rundir}/')
        run.max_power_freq = max_power_freq
        run.max_power = max_power
        run.mnumbers = mnumbers
        runs.append(run)

    for it in range(120, 721, 20):
        fig, axes = plt.subplots(3, 3, figsize=(16,16))
        for irun, run in enumerate(runs):
            draw_equatorial(run, np.log10(run.max_power[..., it]), fig=fig, ax=axes[irun,0],
                            vmin=-6, vmax=-3, gridline=True)
            draw_equatorial(run, run.max_power_freq[..., it], fig=fig, ax=axes[irun,1],
                            vmin=1e-3, vmax=8e-3, gridline=True)
            draw_equatorial(run, np.abs(run.mnumbers[..., it]), fig=fig, ax=axes[irun,2],
                            vmin=0, vmax=20, gridline=True)
        plt.suptitle(f'Time = {(it*5.):.1f} s', fontsize=24)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'wavefig/cwt_analysis_all_it{it:04d}.png')
        plt.close()

def compare_points():
    filenames = ['cwt_analysis_case1_Pc5_Ephi.npz',
                 'cwt_analysis_case2_Pc5_Ephi.npz',
                 'cwt_analysis_case3_Pc5_Ephi.npz']
    rundirs = ['case1b256', 'case2b256new', 'case3b256']

    runs = []
    for filename, rundir in zip(filenames, rundirs):
        max_power_freq, max_power, mnumbers = read_cwt_analysis_data(filename)
        run = Run(f'../../run/{rundir}/')
        run.set_trange((0, 2161, 1))
        run.max_power_freq = max_power_freq
        run.max_power = max_power
        run.mnumbers = mnumbers
        runs.append(run)

    points = [(4, 0), (4, 10), (4, 20)]
    lynestyles = ['solid', 'dashed', 'dotted']

    fig, axes = plt.subplots(4, 1, figsize=(10,12))

    print(run.max_power.shape)

    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, run in enumerate(runs):
    
        for j, point in enumerate(points):
            axes[0].plot(run.time, np.log10(run.max_power[point[1], point[0], :]),
                         color=colors[i], label=f'Case {i+1}, i3 = {point[1]}', linestyle=lynestyles[j])
            axes[1].plot(run.time, run.max_power_freq[point[1], point[0], :],
                         color=colors[i], linestyle=lynestyles[j])
            axes[2].plot(run.time, np.abs(run.mnumbers[point[1], point[0], :]),
                         color=colors[i], linestyle=lynestyles[j])
            axes[3].plot(run.time, run.max_power_freq[point[1], point[0], :] / np.abs(run.mnumbers[point[1], point[0], :]),
                         color=colors[i], linestyle=lynestyles[j])
    axes[0].set_ylabel('Power log10([mV/m]^2)')
    axes[1].set_ylabel('Frequency [Hz]')
    axes[2].set_ylabel('m number')
    axes[3].set_ylabel('Frequency / m number [Hz]')
    axes[3].set_ylim(0, 1e-3)
    axes[3].set_xlabel('Time [s]')
    axes[0].legend()
    for ax in axes:
        ax.grid()
        ax.set_xlim(1000, 3000)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    #compare_equatorial()
    compare_points()
