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
