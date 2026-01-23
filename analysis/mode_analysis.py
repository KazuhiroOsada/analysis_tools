import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from chunk_reader import bg_reader, field_reader
from wavelet import bandpass_filter


def calc_field_aligned_axis(run, i3, i2):
    """
    Parameters
    ----------
    run    : Run object
    i3, i2 : indices in x3 and x2 direction

    Returns
    -------
    s : (N1+1,) array of field-aligned coordinate [Re]
    """
    s = np.zeros(run.N1+1)
    for i1 in range(run.N1//2,0,-1):
        s[i1-1] = s[i1] + run.h1[i3,i2,i1-1]*run.dx1/run.Re
    s[run.N1//2+1:] = -s[run.N1//2-1::-1]
    return s

def draw_x1_time(run, z, i3, i2, fig=None, ax=None, vmax=None, cmap='coolwarm', label=''):
    """
    Draw wave contour plot on x1 time plane at given (i2, i3)

    Parameters
    ----------
    run    : Run object
    z      : (N1, Nt) array
    i3, i2 : indices in x3 and x2 direction
    fig, ax: matplotlib Figure and Axes objects (if None, create new)
    vmax   : max value for color scale
    cmap   : colormap 'coolwarm' in default
    label  : colorbar label
    """
    was_fig_none = fig is None or ax is None
    if was_fig_none:
        fig, ax = plt.subplots()

    s = calc_field_aligned_axis(run, i3, i2)
    dt = run.time[1] - run.time[0]
    t = np.linspace(run.time[0]-dt/2, run.time[-1]+dt/2, run.Nt+1)
    t = t/60
    T, S = np.meshgrid(t, s)

    if vmax is None:
        vmax = np.max(np.abs(z))
    vmin = -vmax

    pcm = ax.pcolormesh(T, S, z, vmin=vmin, vmax=vmax, cmap=cmap)
    s_ticks = np.linspace(np.ceil(s.min()), np.floor(s.max()), 5)
    ax.set_yticks(s_ticks)
    Rh = np.sqrt(run.Xh[i3,i2,:]**2 + run.Yh[i3,i2,:]**2)
    mlat = np.degrees(np.arctan2(run.Zh[i3,i2,:], Rh))
    mlat_ticks = np.interp(s_ticks[::-1], s[::-1], mlat[::-1])[::-1]
    def clean_zero(x):
        return 0.0 if abs(x) < 1e-10 else x
    ytick_labels = [f'{clean_zero(m):.1f}° {s:.1f}' for m, s in zip(mlat_ticks, s_ticks)]
    ax.set_yticklabels(ytick_labels)

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(label, fontsize=16)
    cbar.ax.ticklabel_format(style='sci', scilimits=(0,0))
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.yaxis.get_offset_text().set_fontsize(14)

    if was_fig_none:
        plt.show()


if __name__ == '__main__':
    i2, i3 = 4, 5
    trange = (0, 1441, 1)

    rundirs = ['../../run/case1b128',
               '../../run/case2b128',
               '../../run/case3b128']
    
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    case_labels = ['Case 1', 'Case 2', 'Case 3']

    fig, axes = plt.subplots(9, 1, figsize=(12, 18), sharex=True)  
    for i, run_dir in enumerate(rundirs):
        run = Run(run_dir)
        run.read('coord')
        run.set_trange(trange)
        _, d2, d3, _, l2, l3 = run.resolve_global_idx(0, i2, i3)

        B0 = np.zeros((run.N3_local, run.N2_local, run.N1, 3))
        B = np.zeros((run.N3_local, run.N2_local, run.N1, 3, run.Nt))
        V = np.zeros((run.N3_local, run.N2_local, run.N1, 3, run.Nt))
        for d1 in range(run.domain[2]):
            filepath_bg = f'{run.prefix}/bg-{d1:02d}-{d2:02d}-{d3:02d}.dat'
            B0[..., d1*run.N1_local:(d1+1)*run.N1_local, :], _ = bg_reader(filepath_bg, run.N1_local, run.N2_local, run.N3_local)
            filepath_field = f'{run.prefix}/field-{d1:02d}-{d2:02d}-{d3:02d}.dat'
            V[..., d1*run.N1_local:(d1+1)*run.N1_local, :, :], B[..., d1*run.N1_local:(d1+1)*run.N1_local, :, :] = field_reader(filepath_field, run.N1_local, run.N2_local, run.N3_local, trange)
    
        Btot = B + B0[..., None]
        E = -np.cross(V, Btot, axis=3)

        s = calc_field_aligned_axis(run, i2, i3)
     
        low_cutoff = 2.0e-3
        high_cutoff = 7.0e-3
        dt = run.delt * run.ifdiag

        Ephi  = bandpass_filter(run, E[..., 2, :], low_cutoff, high_cutoff) * run.unitE
        Br    = bandpass_filter(run, -B[..., 1, :], low_cutoff, high_cutoff) * run.unitB
        Bpara = bandpass_filter(run, -B[..., 0, :], low_cutoff, high_cutoff) * run.unitB

        axes[i*3+1].text(-0.16, 0.5, case_labels[i], color='black', fontsize=18,
                         transform=axes[i*3+1].transAxes, rotation=90, va='center')

        axes[i*3+0].text(-0.13, 0.9, labels[i*3+0], color='black', fontsize=16, fontweight='bold', transform=axes[i*3+0].transAxes)
        draw_x1_time(run, Ephi[l3, l2, :, :], i2, i3, fig=fig, ax=axes[i*3+0], vmax=5e-4, label='$E_\phi$ [mV/m]')
        axes[i*3+1].text(-0.13, 0.9, labels[i*3+1], color='black', fontsize=16, fontweight='bold', transform=axes[i*3+1].transAxes)
        draw_x1_time(run, Br[l3, l2, :, :], i2, i3, fig=fig, ax=axes[i*3+1], vmax=2e-3, label='$B_r$ [nT]')
        axes[i*3+2].text(-0.13, 0.9, labels[i*3+2], color='black', fontsize=16, fontweight='bold', transform=axes[i*3+2].transAxes)
        draw_x1_time(run, Bpara[l3, l2, :, :], i2, i3, fig=fig, ax=axes[i*3+2], vmax=2e-3, label='$B_{||}$ [nT]')
    axes[0].text(-0.1, 1.1, 'MLAT [°] s [Re]', fontsize=12, transform=axes[0].transAxes)
    axes[8].set_xlabel('Time [min]', fontsize=16)
    axes[8].set_xlim(0, 60)
    plt.tight_layout()
    plt.savefig('mode_analysis.pdf')
