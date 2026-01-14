import os
import sys
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from chunk_reader import bg_reader, field_reader
from analysis.wavelet import bandpass_filter
from analysis.mode_analysis import draw_x1_time


savefiles = [f'mode_analysis_case{irun+1}.pdf' for irun in range(3)]

run_prefix = '../../../run'
rundirs = [os.path.join(run_prefix, f'case{irun+1}b128') for irun in range(3)]

i3, i2 = 5, 4

from time import time
def create_figure(rundir, savefile):
    run = Run(rundir)
    run.read('coord')
    trange = (0, 1441, 1)
    run.set_trange(trange)
    _, d2, d3, _, l2, l3 = run.resolve_global_idx(0, i2, i3)

    run.L = 1 / run.x2**2 / run.Re

    B0 = np.zeros((run.N3_local, run.N2_local, run.N1, 3))
    B = np.zeros((run.N3_local, run.N2_local, run.N1, 3, run.Nt))
    V = np.zeros((run.N3_local, run.N2_local, run.N1, 3, run.Nt))
    for d1 in range(run.domain[2]):
        filepath_bg = os.path.join(run.prefix, f'bg-{d1:02d}-{d2:02d}-{d3:02d}.dat')
        B0[..., d1*run.N1_local:(d1+1)*run.N1_local, :], _ = bg_reader(filepath_bg, run.N1_local, run.N2_local, run.N3_local)
        filepath_field = os.path.join(run.prefix, f'field-{d1:02d}-{d2:02d}-{d3:02d}.dat')
        V[..., d1*run.N1_local:(d1+1)*run.N1_local, :, :], B[..., d1*run.N1_local:(d1+1)*run.N1_local, :, :] = field_reader(filepath_field, run.N1_local, run.N2_local, run.N3_local, trange)
    B0 = B0[l3, l2, :, :] * run.unitB
    B = B[l3, l2, :, :, :] * run.unitB
    V = V[l3, l2, :, :, :] * run.unitV
    Btot = B + B0[..., None]
    E = -np.cross(V, Btot, axis=1) / run.unitV / run.unitB * run.unitE

    low_cutoff = 2.0e-3
    high_cutoff = 7.0e-3
    Bpara = bandpass_filter(run, -B[:, 0, :], low_cutoff, high_cutoff)
    # transform to spherical coordinate
    r = np.sqrt(run.Xi[i3, i2, :]**2 + run.Yi[i3, i2, :]**2 + run.Zi[i3, i2, :]**2)
    cost = run.Zi[i3, i2, :] / r
    sint = np.sqrt(1 - cost**2)
    gam = np.sqrt(1 + 3*cost**2)
    R11 = (2*cost / gam)[:, None]
    R12 = (-sint / gam)[:, None]
    Br = R11 * B[:, 0, :] + R12 * B[:, 1, :]
    Ephi = E[:, 2, :]
    Br = bandpass_filter(run, Br, low_cutoff, high_cutoff)
    Ephi = bandpass_filter(run, Ephi, low_cutoff, high_cutoff)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    labels = ['a', 'b', 'c']

    axes[0].text(-0.18, 1.1, 'MLAT [°] s [Re]', fontsize=14, transform=axes[0].transAxes)
    axes[0].set_title(f'Case {run.prefix.name[4]}, L = {round(run.L[i2],1)}, MLT = 13 h', fontsize=16)
    axes[0].text(-0.2, 1.0, labels[0], color='black', fontsize=18, fontweight='bold', transform=axes[0].transAxes)
    axes[0].tick_params(labelsize=14)
    draw_x1_time(run, Ephi, i3, i2, fig=fig, ax=axes[0], vmax=5e-4, label='$E_\phi$ [mV/m]')
    axes[1].text(-0.2, 1.0, labels[1], color='black', fontsize=18, fontweight='bold', transform=axes[1].transAxes)
    axes[1].tick_params(labelsize=14)
    draw_x1_time(run, Br, i3, i2, fig=fig, ax=axes[1], vmax=1.5e-3, label='$B_r$ [nT]')
    axes[2].text(-0.2, 1.0, labels[2], color='black', fontsize=18, fontweight='bold', transform=axes[2].transAxes)
    draw_x1_time(run, Bpara, i3, i2, fig=fig, ax=axes[2], vmax=1.5e-3, label='$B_{||}$ [nT]')
    axes[2].tick_params(labelsize=14)
    axes[2].set_xlabel('Time [min]', fontsize=16)
    axes[2].set_xlim(0, 60)

    plt.savefig(savefile, bbox_inches='tight')

for rundir, savefile in zip(rundirs, savefiles):
    create_figure(rundir, savefile)
