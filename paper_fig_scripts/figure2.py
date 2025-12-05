import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.collections import QuadMesh

from base import Run
from draw import draw_equatorial
from analysis.dps import calc_SYM_H, calc_total_energy


savefile = 'symh.pdf'

rundirs = ['../../run/case1b128',
           '../../run/case2b128',
           '../../run/case3b128']
runs = []

for i, rundir in enumerate(rundirs):
    run = Run(rundir)
    run.set_trange((0, 1441, 120))
    run.read('coord')
    run.read('moment')
    print(f'data read completed for case {i+1}')
    run.total_energy = calc_total_energy(run)
    run.sym_h = calc_SYM_H(run)
    print(f'calculation completed for case {i+1}')
    runs.append(run)

fig = plt.figure(figsize=(13, 12))
outer = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.15)
top = GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[0],
                                hspace=0.1, wspace=0.1,
                                width_ratios=[1, 1, 1, 0.3])
bottom = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], hspace=0.3)

ts_min = [60, 20]

label_x, label_y = -0.2, 1.15
fs_label = 25
labels_top = [['(a)', '(b)'], ['(c)', '(d)'], ['(e)', '(f)']]
labels_top = [['', ''], ['', ''], ['', '']]

for irun, run in enumerate(runs):
    for i, t_min in enumerate(ts_min):
        it = round(t_min*60 / (run.delt*run.ifdiag*run.trange[2]))
        ax = fig.add_subplot(top[i, irun])
        ax.text(label_x, label_y, f'$\mathbf{{{labels_top[irun][i]}}}$',
                transform=ax.transAxes, fontsize=fs_label, va='top')
        draw_equatorial(run, run.Ppe[..., run.N1//2, it], fig=fig, ax=ax,
                        title=f'Case {irun+1}' if i==0 else None, vmin=1e-4, vmax=10, log=True,
                        colorbar=False, xlabel=(i==1), ylabel=(irun==0), width=8.0)
        pos = ax.get_position()
        if irun == 0:
            fig.text(pos.x0 - 0.08, pos.y0 + pos.height/6, f't = {t_min} min',
                    ha='center', va='bottom', fontsize=fs_label, rotation=90)
        if irun == 2:
            cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
            pcm = next(c for c in ax.get_children() if isinstance(c, QuadMesh))
            cbar = fig.colorbar(pcm, cax=cbar_ax)
            cbar.set_label('$P_\perp$ [nPa]', fontsize=fs_label)

label_x, label_y = -0.1, 1.2
labels_bottom = ['(g)', '(h)']
labels_bottom = ['', '']
fs_ticks = 16

ax1 = fig.add_subplot(bottom[0, 0])
ax1.text(label_x, label_y, f'$\mathbf{{{labels_bottom[0]}}}$',
            transform=ax1.transAxes, fontsize=fs_label, va='top')
ax1.set_ylabel('SYM-H\n index\n [nT]', fontsize=fs_label)
ax1.set_ylim(-20, 0)
ax2 = fig.add_subplot(bottom[1, 0])
ax2.text(label_x, label_y, f'$\mathbf{{{labels_bottom[1]}}}$',
            transform=ax2.transAxes, fontsize=fs_label, va='top')
ax2.set_ylabel('Total energy\nof ring\n current\n [J]', fontsize=fs_label)
ax2.set_ylim(0, 6e14)
ax2.set_xlabel('Time [min]', fontsize=fs_label)
for ax in [ax1, ax2]:
    ax.set_xlim(0, 120)
    ax.tick_params(labelsize=fs_ticks)
    ax.minorticks_on()
    ax.grid()
    ax.axvline(x=ts_min[0], color='k', linestyle='--')
    ax.axvline(x=ts_min[1], color='k', linestyle='--')
for run in runs:
    ax1.plot(run.time/60, run.sym_h, label=f'Case {runs.index(run)+1}')
    ax2.plot(run.time/60, run.total_energy)
ax1.legend(fontsize=fs_ticks)

plt.savefig(savefile, format='pdf', dpi=300, bbox_inches='tight')
plt.show()
