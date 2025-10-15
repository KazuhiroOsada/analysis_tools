import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from chunk_reader import bg_reader, field_reader
from wavelet import bandpass_filter


def calc_field_aligned_axis(run, i2, i3):
    """
    arguments: run -- Run object
    return: s -- (N1+1,) field aligned coordinate for pcolormesh [Re]
    """
    s = np.zeros(run.N1+1)
    for i1 in range(run.N1//2,0,-1):
        s[i1-1] = s[i1] + run.h1[i3,i2,i1-1]*run.dx1/run.Re
    s[run.N1//2+1:] = -s[run.N1//2-1::-1]
    return s

def draw_x1_time(run, z, i2, i3, fig=None, ax=None, vmax=None, cmap='coolwarm', label=''):
    """
    arguments: run -- Run object
               z   -- (N1, Nt) array
    """
    was_fig_none = fig is None or ax is None
    if was_fig_none:
        fig, ax = plt.subplots()

    s = calc_field_aligned_axis(run, i2, i3)
    dt = run.time[1] - run.time[0]
    t = np.linspace(run.time[0]-dt/2, run.time[-1]+dt/2, run.Nt+1)
    T, S = np.meshgrid(t, s)

    if vmax is None:
        vmax = np.max(np.abs(z))
    vmin = -vmax

    pcm = ax.pcolormesh(T, S, z, vmin=vmin, vmax=vmax, cmap=cmap)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(label)
    s_ticks = np.linspace(np.ceil(s.min()), np.floor(s.max()), 5)
    ax.set_yticks(s_ticks)
    Rh = np.sqrt(run.Xh[i3,i2,:]**2 + run.Yh[i3,i2,:]**2)
    mlat = np.degrees(np.arctan2(run.Zh[i3,i2,:], Rh))
    mlat_ticks = np.interp(s_ticks[::-1], s[::-1], mlat[::-1])[::-1]
    def clean_zero(x):
        return 0.0 if abs(x) < 1e-10 else x
    ytick_labels = [f'{clean_zero(m):.1f}° {s:.1f}' for m, s in zip(mlat_ticks, s_ticks)]
    ax.set_yticklabels(ytick_labels)

    if was_fig_none:
        plt.show()


if __name__ == '__main__':
    i2, i3 = 4, 20
    trange = (0, 2161, 1)

    run = Run('../../run/case1b256')
    run.read('coord')
    run.set_trange(trange)

    d3, l3 = i3 // run.N3_local, i3 % run.N3_local
    d2, l2 = i2 // run.N2_local, i2 % run.N2_local

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

    low_cutoff = 1.6e-3
    high_cutoff = 7e-3
    dt = run.delt * run.ifdiag

    Ephi  = bandpass_filter(run, E[..., 2, :], low_cutoff, high_cutoff) * run.unitE
    Er    = bandpass_filter(run, E[..., 1, :], low_cutoff, high_cutoff) * run.unitE
    Br    = bandpass_filter(run, B[..., 1, :], low_cutoff, high_cutoff) * run.unitB
    Bphi  = bandpass_filter(run, B[..., 2, :], low_cutoff, high_cutoff) * run.unitB
    Bpara = bandpass_filter(run, B[..., 0, :], low_cutoff, high_cutoff) * run.unitB

    fig, ax = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    draw_x1_time(run, Ephi[l3, l2, :, :], i2, i3, fig=fig, ax=ax[0], label='Ephi [mV/m]')
    draw_x1_time(run, Er[l3, l2, :, :], i2, i3, fig=fig, ax=ax[1], label='Er [mV/m]')
    draw_x1_time(run, Br[l3, l2, :, :], i2, i3, fig=fig, ax=ax[2], label='Br [nT]')
    draw_x1_time(run, Bphi[l3, l2, :, :], i2, i3, fig=fig, ax=ax[3], label='Bphi [nT]')
    draw_x1_time(run, Bpara[l3, l2, :, :], i2, i3, fig=fig, ax=ax[4], label='Bpara [nT]')
    ax[2].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.show()
