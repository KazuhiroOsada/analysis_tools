import sys
import time
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from draw import draw_equatorial


def moving_average_nan(time, data):
    window = 10
    n = len(data)
    result = np.full(n, np.nan)

    start = np.argmax(~np.isnan(data))
    end = n - 1 - np.argmax(~np.isnan(data[::-1]))

    core = data[start:end+1]
    filtered = np.zeros_like(core, dtype=float)

    half = window // 2
    for i in range(len(core)):
        s = max(0, i - half)
        e = min(len(core), i + half + 1)
        filtered[i] = np.nanmean(core[s:e])

    result[start:end+1] = filtered
    return time, result

def read_cwt_analysis_data(cwt_analysis_file):
    data = np.load(cwt_analysis_file)
    max_power_freq = data['max_power_freq']
    max_power      = data['max_power']
    mnumbers       = data['mnumbers']
    time           = data['time']
    return max_power_freq, max_power, mnumbers, time

def draw_one_case(rundir, cwt_analysis_file, trange=(0, 1441, 20)):
    max_power_freq, max_power, mnumbers, time = read_cwt_analysis_data(cwt_analysis_file)
    run = Run(rundir)
    run.read_equatorial('coord')
    run.set_trange(trange)
    run.read_equatorial('moment')

    for it in range(run.Nt):
        fig, axes = plt.subplots(1, 4, figsize=(16,6))
        draw_equatorial(run, run.Ppe[..., it], fig=fig, ax=axes[0],
                        vmin=1e-4, vmax=10, log=True, title='Pperp [nPa]')
        draw_equatorial(run, np.log10(max_power[..., it*trange[2]]), fig=fig, ax=axes[1],
                        vmin=-6, vmax=-4, gridline=True)
        draw_equatorial(run, max_power_freq[..., it*trange[2]], fig=fig, ax=axes[2],
                        vmin=1e-3, vmax=8e-3, gridline=True)
        draw_equatorial(run, mnumbers[..., it*trange[2]], fig=fig, ax=axes[3],
                        vmin=-20, vmax=20, cmap='coolwarm', gridline=True)
        plt.suptitle(f'Time = {time[it*trange[2]]:.1f} s', fontsize=24)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'fig/{run.prefix.name}_it{it:04d}.png')
        plt.close()

def compare_equatorial():
    """
    Draw equatorial profiles of cwt analysis results for multiple runs
    """
    filenames = ['cwt/analysis_case1b128_Ephi.npz',
                 'cwt/analysis_case2b128_Ephi.npz',
                 'cwt/analysis_case3b128_Ephi.npz']
    rundirs = ['case1b128', 'case2b128', 'case3b128']

    runs = []
    for filename, rundir in zip(filenames, rundirs):
        max_power_freq, max_power, mnumbers, time = read_cwt_analysis_data(filename)
        mask = max_power_freq < 2.0e-3
        max_power_freq[mask] = np.nan
        max_power[mask] = np.nan
        mnumbers[mask] = np.nan

        run = Run(f'../../run/{rundir}/')
        run.read_equatorial('coord')
        run.set_trange((0, 400, 1))
        run.read_equatorial('moment')
        run.max_power_freq = max_power_freq
        run.max_power = max_power
        run.mnumbers = mnumbers
        runs.append(run)

    pcms = [None]*4
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    for it in range(300, 301, 1):
        fig, axes = plt.subplots(3, 4, figsize=(16,13))
        for irun, run in enumerate(runs):
            pcms[0] = draw_equatorial(run, run.Ppe[..., it], fig=fig, ax=axes[irun,0],
                            vmin=1e-4, vmax=10, log=True, xlabel=False, ylabel=True,
                            title='P$_\perp$ [nPa]' if irun==0 else None, colorbar=False, return_pcm=True, width=8)
            pcms[1] = draw_equatorial(run, np.log10(run.max_power[..., it]), fig=fig, ax=axes[irun,1],
                            vmin=-6.5, vmax=-4.5, gridline=True,  xlabel=False, ylabel=False,
                            title='log$_{10}$(Power [(mV/m)$^2$])' if irun==0 else None, colorbar=False, return_pcm=True, width=8)
            pcms[2] = draw_equatorial(run, run.max_power_freq[..., it]*1e3, fig=fig, ax=axes[irun,2],
                            vmin=2, vmax=8, gridline=True,  xlabel=False, ylabel=False,
                            title='Frequency [mHz]' if irun==0 else None, colorbar=False, return_pcm=True, width=8)
            pcms[3] = draw_equatorial(run, run.mnumbers[..., it], fig=fig, ax=axes[irun,3],
                            vmin=-20, vmax=20, gridline=True, cmap='coolwarm',  xlabel=False, ylabel=False,
                            title='m number' if irun==0 else None, colorbar=False, return_pcm=True, width=8)
        for i in range(4):
            ax = axes[2, i]
            pos = ax.get_position()
            cax = fig.add_axes([pos.x0, pos.y0 - 0.024 - 0.018, pos.width, 0.018])
            fig.colorbar(pcms[i], cax=cax, orientation="horizontal")
            
        #plt.suptitle(f'Time = {time[it]} s', fontsize=24)
        plt.savefig('eq_summary.pdf')

def compare_points():
    """
    Compare cwt analysis results at specific points for multiple runs
    """
    filenames = ['cwt/analysis_case1b128_Ephi.npz',
                 'cwt/analysis_case2b128_Ephi.npz',
                 'cwt/analysis_case3b128_Ephi.npz']
    rundirs = ['case1b128', 'case2b128', 'case3b128']

    runs = []
    for filename, rundir in zip(filenames, rundirs):
        max_power_freq, max_power, mnumbers, time = read_cwt_analysis_data(filename)
        mask = time < 780
        if rundir == 'case2b128':
            mask = (time < 780) | (time > 2210)
        max_power_freq[:, :, mask] = np.nan
        max_power[:, :, mask] = np.nan
        mnumbers[:, :, mask] = np.nan

        run = Run(f'../../run/{rundir}/')
        run.read_equatorial('coord')
        run.set_trange((0, 1441, 1))
        run.read_equatorial('moment')
        run.max_power_freq = max_power_freq
        run.max_power = max_power
        run.mnumbers = mnumbers
        runs.append(run)

    i3, i2 = 4, 4

    from growthrate import calc_gammas
    for run, cwt_file in zip(runs, filenames):
        run.set_trange((0, 101, 1), 'v')
        run.gss = []
        run.res_Ws = []
        run.time_g, _, _, gs, vpe, vpa, _ = calc_gammas(run, cwt_file, i3, i2, plot=False)
        np.savez(f'gamma_{run.prefix.name}_i3{i3}_i2{i2}.npz', time=run.time_g, gs=gs)
        Ws = 1/2*run.Mp*(vpe**2 + vpa**2) / run.Qp *1e3
        if run.prefix.name == 'case2b128':
            # for case2b128, only keep data between 13 min and 36.8 min
            mask = run.time_g > 2210
            Ws[mask] = np.nan
            gs[mask] = np.nan
        run.gss.append(gs)
        run.res_Ws.append(Ws)  # in kV
        

    linestyles = ['solid', 'dashed', 'dotted']
    fig, axes = plt.subplots(5, 1, figsize=(10,10), sharex=True)
    colors = ['black', 'red', 'blue']
    labels = ['Case 1', 'Case 2', 'Case 3']
    ch_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
    for i, ch in enumerate(ch_labels):
        axes[i].text(-0.15, 1.0, ch, color='black', fontsize=14, fontweight='bold', transform=axes[i].transAxes)
    for i, run in enumerate(runs):
        j = 0  # only one point for now
        axes[0].plot(run.time/60, np.log10(run.max_power[i3, i2, :]),
                        color=colors[i], linestyle=linestyles[j], label=labels[i],)
        axes[0].set_ylabel('log$_{10}$(Power [(mV/m)$^2$])', fontsize=12)
        axes[1].plot(run.time/60, run.max_power_freq[i3, i2, :]*1e3,
                        color=colors[i], linestyle=linestyles[j], label=labels[i],)
        axes[1].set_ylabel('Frequency [mHz]', fontsize=12)
        axes[2].plot(run.time/60, run.mnumbers[i3, i2, :],
                        color=colors[i], linestyle=linestyles[j], label=labels[i],)
        axes[2].set_ylabel('m number', fontsize=12)
        t_ma, gs_ma = moving_average_nan(run.time_g/60, run.gss[j])
        axes[3].plot(t_ma, gs_ma, '.-',
                        color=colors[i], linestyle=linestyles[j], label=labels[i])
        axes[3].set_ylabel('Growth rate [1/s]', fontsize=12)
        axes[3].plot(run.time_g/60, run.gss[j], '.-',
                        color=colors[i], linestyle=linestyles[j], alpha=0.3)
        t_ma, res_Ws_ma = moving_average_nan(run.time_g/60, run.res_Ws[j])
        axes[4].plot(t_ma, res_Ws_ma, '.-',
                     color=colors[i], linestyle=linestyles[j], label=labels[i])
        axes[4].set_ylabel('W$_\mathrm{resonant}$[kV]', fontsize=12)
        axes[4].plot(run.time_g/60, run.res_Ws[j],
                     color=colors[i], linestyle=linestyles[j], alpha=0.3)
        axes[4].set_xlabel('Time [min]', fontsize=12)
        axes[4].set_ylim(0, 125)
    axes[0].legend(fontsize=12)
    for ax in axes:
        ax.grid()
        ax.set_xlim(960/60, 2400/60)
    plt.tight_layout()
    plt.savefig('cwt_point_comparison.pdf')
    plt.show()

def compare_points_manual():
    filenames = ['cwt/analysis_case1b128_Ephi.npz',
                 'cwt/analysis_case2b128_Ephi.npz',
                 'cwt/analysis_case3b128_Ephi.npz']
    rundirs = ['case1b128', 'case2b128', 'case3b128']

    runs = []
    for filename, rundir in zip(filenames, rundirs):
        max_power_freq, max_power, mnumbers, _ = read_cwt_analysis_data(filename)

        run = Run(f'../../run/{rundir}/')
        run.read_equatorial('coord')
        run.set_trange((0, 1441, 1))
        run.read_equatorial('moment')


if __name__ == '__main__':
    compare_points()
    #draw_one_case('../../run/case1b128', 'cwt/analysis_case1b128_Ephi.npz')
