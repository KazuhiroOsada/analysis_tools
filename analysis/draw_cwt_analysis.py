import sys
import time
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from draw import draw_equatorial


def moving_average_nan(time, data):
    window = 20
    n = len(data)
    result = np.full(n, np.nan)

    # --- 実データが出てくる最初の index ---
    start = np.argmax(~np.isnan(data))
    # --- 実データが最後にある index ---
    end = n - 1 - np.argmax(~np.isnan(data[::-1]))

    # 実データ部分だけ取り出す
    core = data[start:end+1]

    # core 部分の移動平均（NaN 無視）
    filtered = np.zeros_like(core, dtype=float)

    half = window // 2

    for i in range(len(core)):
        # ウィンドウの範囲
        s = max(0, i - half)
        e = min(len(core), i + half + 1)
        filtered[i] = np.nanmean(core[s:e])

    # 結果に戻す
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
        run.set_trange((0, 1441, 1))
        run.read_equatorial('moment')
        run.max_power_freq = max_power_freq
        run.max_power = max_power
        run.mnumbers = mnumbers
        runs.append(run)

    for it in range(200, 401, 5):
        fig, axes = plt.subplots(3, 4, figsize=(16,12))
        for irun, run in enumerate(runs):
            draw_equatorial(run, run.Ppe[..., it], fig=fig, ax=axes[irun,0],
                            vmin=1e-4, vmax=10, log=True)
            draw_equatorial(run, np.log10(run.max_power[..., it]), fig=fig, ax=axes[irun,1],
                            vmin=-4.5, vmax=-7.5, gridline=True)
            draw_equatorial(run, run.max_power_freq[..., it], fig=fig, ax=axes[irun,2],
                            vmin=1e-3, vmax=8e-3, gridline=True)
            draw_equatorial(run, run.mnumbers[..., it], fig=fig, ax=axes[irun,3],
                            vmin=-20, vmax=20, gridline=True, cmap='coolwarm')
        plt.suptitle(f'Time = {time[it]} s', fontsize=24)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'cwtfig/cwt_analysis_all_it{it:04d}.png')
        plt.close()

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
        run.time_g, _, _, gs, _, _, _ = calc_gammas(run, cwt_file, i3, i2, plot=True)
        np.savez(f'gamma_{run.prefix.name}_i3{i3}_i2{i2}.npz', time=run.time_g, gs=gs)
        run.gss.append(gs)

    linestyles = ['solid', 'dashed', 'dotted']
    fig, axes = plt.subplots(5, 1, figsize=(10,14), sharex=True)
    colors = ['black', 'red', 'blue']
    for i, run in enumerate(runs):
        j = 0  # only one point for now
        axes[0].plot(run.time, np.log10(run.max_power[i3, i2, :]),
                        color=colors[i], linestyle=linestyles[j])
        axes[1].plot(run.time, run.max_power_freq[i3, i2, :],
                        color=colors[i], linestyle=linestyles[j])
        axes[2].plot(run.time, run.mnumbers[i3, i2, :],
                        color=colors[i], linestyle=linestyles[j])
        t_ma, gs_ma = moving_average_nan(run.time_g, run.gss[j])
        axes[3].plot(t_ma, gs_ma, '.-',
                        color=colors[i], linestyle=linestyles[j])
        axes[3].plot(run.time_g, run.gss[j], '.-',
                        color=colors[i], linestyle=linestyles[j], alpha=0.3)
        axes[4].plot(run.time, run.Ppe[i3, i2, :],
                        color=colors[i], linestyle=linestyles[j])

    for ax in axes:
        ax.grid()
        ax.set_xlim(750, 2500)
    plt.tight_layout()
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
