import os
import sys
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from chunk_reader import dist_reader
from analysis.draw_psd import draw_on_velocity_space
from analysis.growthrate import calc_gamma_velocity_space, calc_bounce_and_drift_freqs, calc_vperp


savefiles = [f'gamma_velocity_space_case{irun+1}.pdf' for irun in range(3)]

run_prefix = '../../../run'
rundirs = [os.path.join(run_prefix, f'case{irun+1}b128') for irun in range(3)]
cwt_analysis_prefix = '../../analysis/processed/analysis'
cwt_analysis_files = [os.path.join(cwt_analysis_prefix, f'case{irun+1}b128.npz') for irun in range(3)]

i3, i2 = 5, 4
trange = (24, 37, 12) # 20, 30 min

def create_figure(rundir, savefile, cwt_file):
    fig, axes = plt.subplots(2, 3, figsize=(10,12))

    run = Run(rundir)
    run.read_equatorial('coord')
    run.set_trange(trange, 'v')

    data = np.load(cwt_file)
    it_cwt = np.rint(run.time_v / (run.ifdiag*run.delt)).astype(int)
    freq = data['max_power_freq']
    mnum = data['mnumbers']

    d1, d2, d3, l1, l2, l3 = run.resolve_global_idx(run.N1//2, i2, i3)
    assert(l2 != 0 and l2 != run.N2_local - 1)
    dist_file = os.path.join(run.prefix, f'dist1-{d1:02d}-{d2:02d}-{d3:02d}.dat')
    dist_chunk = dist_reader(dist_file, run.N1_local, run.N2_local, run.N3_local, run.Nm, run.Nv, run.trange_v)

    run.read_equatorial('bg')
    run.read_equatorial('field')
    run.calc_magnetic_amplitude()  

    for it in range(2):
        dist_m = dist_chunk[l3, l2-1, l1, :, :, it]
        dist   = dist_chunk[l3, l2,   l1, :, :, it]
        dist_p = dist_chunk[l3, l2+1, l1, :, :, it]
        B_m = run.Babs[i3, i2-1, it] / run.unitB # T
        B = run.Babs[i3, i2,   it] / run.unitB # T
        B_p = run.Babs[i3, i2+1, it] / run.unitB # T
        omega = freq[i3, i2, it_cwt[it]] * 2*np.pi # rad/s
        m     = mnum[i3, i2, it_cwt[it]]

        gamma1, _, gamma, _ = calc_gamma_velocity_space(run, dist_m, dist, dist_p, B_m, B, B_p, i2, omega, m, n=0)

        pcms = np.zeros(3, dtype=object)
        pcms[0] = draw_on_velocity_space(run, dist, xaxis='vperp', B=B*run.unitB, fig=fig, ax=axes[it, 0], 
                                         log=True, vmin=1e-19, vmax=1e-16, colorbar=False, return_pcm=True, 
                                         title='PSD [s$^3$/m$^6$]' if it==0 else '',
                                         xlabel=it==1, ylabel=True)
        pcms[1] = draw_on_velocity_space(run, gamma, xaxis='vperp', B=B*run.unitB, fig=fig, ax=axes[it, 1], 
                                         cmap='coolwarm', vmin=-0.003, vmax=0.003, colorbar=False, return_pcm=True, 
                                         title='$\gamma$ [/s]$= \gamma_1 + \gamma_2$' if it==0 else '',
                                         xlabel=it==1, ylabel=False)
        pcms[2] = draw_on_velocity_space(run, gamma1, xaxis='vperp', B=B*run.unitB, fig=fig, ax=axes[it, 2], 
                                         cmap='coolwarm', vmin=-0.003, vmax=0.003, colorbar=False, return_pcm=True, 
                                         title='$\gamma_1$ [/s]$\propto df/dW$' if it==0 else '',
                                         xlabel=it==1, ylabel=False)

        for i in range(3):
            _, wd = calc_bounce_and_drift_freqs(run, B, i2)
            vperp = calc_vperp(run, B) * 1e-3 # to km/s
            x, y = np.meshgrid(vperp, run.vp*1e-3, indexing='ij')
            ctr = axes[it, i].contour(x, y, wd*1e3/(2*np.pi), colors='gray', linestyles='dashed')
            axes[it, i].clabel(ctr, fmt='%.2f', colors='gray', fontsize=10) 
            ctr = axes[it, i].contour(x, y, wd*1e3/(2*np.pi), levels=[omega/m*1e3/(2*np.pi)], colors='black', linestyles='dashed', linewidths=2)
            axes[it, i].clabel(ctr, fmt='%.2f', colors='black', fontsize=10)

    for i in range(3):
        pos = axes[1, i].get_position()
        cbar_height = 0.018
        pad = 0.05
        cbar_ax = fig.add_axes([pos.x0, pos.y0 - pad - cbar_height, pos.width, cbar_height])
        fig.colorbar(pcms[i], cax=cbar_ax, orientation='horizontal')
        cbar_ax.tick_params(axis='x', labelsize=14)
        if i >= 1:
            cbar_ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,-3))
            cbar_ax.xaxis.get_offset_text().set_fontsize(14)
        cbar_ax.set_xlabel(['PSD [s$^3$/m$^6$]', '$\gamma$ [/s]', '$\gamma_1$ [/s]'][i], fontsize=18)
    
    labels = [['a', 'b', 'c'], ['d', 'e', 'f']]
    for i in range(2):
        for j in range(3):
            axes[i, j].text(-0.13, 1.0, labels[i][j], color='black', fontsize=20, fontweight='bold', transform=axes[i, j].transAxes)
            axes[i, j].tick_params(axis='x', labelsize=14)
            axes[i, j].tick_params(axis='y', labelsize=14)

    axes[0,0].text(-0.63, 0.5, 't = 20 min', transform=axes[0,0].transAxes,
                    fontsize=20, va='center', ha='center', rotation=90)
    axes[1,0].text(-0.63, 0.5, 't = 30 min', transform=axes[1,0].transAxes,
                    fontsize=20, va='center', ha='center', rotation=90)       

    plt.savefig(savefile, bbox_inches='tight')
    plt.close(fig)

for irun, (rundir, savefile, cwt_file) in enumerate(zip(rundirs, savefiles, cwt_analysis_files)):
    create_figure(rundir, savefile, cwt_file)
