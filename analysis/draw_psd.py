import sys
sys.path.append('..')
import os

import numpy as np
import matplotlib.pyplot as plt

from base import Run


# Note : it is assumed that these functions are assumed to be used when single species is simulated

def convert_mu_to_vperp(mu, B):
    """
    Parameters
    ----------
    mu      : magnetic moment [eV/T]
    B       : magnetic field [nT]

    Returns
    -------
    vperp : perpendicular velocity [m/s]
    """
    Qp = 1.602e-19 # proton charge [C]
    Mp = 1.673e-27 # proton mass [kg]
    return np.sqrt(2 * mu * (B*1e-9) * Qp / Mp)

def draw_on_velocity_space(run, Z, xaxis='mu', B=None, fig=None, ax=None,
                           log=False, vmin=None, vmax=None,
                           xlabel=True, ylabel=True,
                           cmap='viridis', alpha=1.0, colorbar=True,
                           label='', title=None, savefile=None, return_pcm=False):
    """
    Parameters
    ----------
    run        : Run object
    Z          : 2D array to be drawn on velocity space, shape=(Nm, Nv)
    xaxis      : 'mu' or 'vperp' (default: 'mu')
    B          : magnetic field [nT] (required when xaxis='vperp')
    fig, ax    : matplotlib figure and axis objects (default: None, create new)
    log        : whether to use logarithmic scale for color map (default: False)
    vmin, vmax : min and max values for color map (default: None, use min/max of Z)
    cmap       : colormap (default: 'viridis')
    alpha      : alpha value for color map (default: 1.0)
    colorbar   : whether to draw colorbar (default: True)
    label      : label for colorbar (default: '')
    title      : title for the plot (default: '')
    savefile   : path to save the figure (default: None, show on screen)
    """
    was_fig_none = fig is None or ax is None
    if was_fig_none:
        fig, ax = plt.subplots(figsize=(8, 8))

    log_dm = (np.log10(run.mu[-1]) - np.log10(run.mu[0])) / (run.Nm - 1)
    x = 10**np.linspace(np.log10(run.mu[0]) - log_dm/2, np.log10(run.mu[-1]) + log_dm/2, run.Nm+1)
    dv = (run.vp[-1] - run.vp[0]) / (run.Nv - 1)
    y = np.linspace(run.vp[0] - dv/2, run.vp[-1] + dv/2, run.Nv+1) * 1e-3 # to km/s

    if xaxis == 'mu':
        x *= 1e-12 # eV/nT to keV/nT
        X, Y = np.meshgrid(x, y, indexing='ij')
        ax.set_xscale('log')
        xlabel_text = 'mu [keV/nT]'
    elif xaxis == 'vperp':
        if B is None:
            print('argument B is required when xaxis="vperp"')
            return
        x = convert_mu_to_vperp(x, B) * 1e-3 # to km/s
        X, Y = np.meshgrid(x, y, indexing='ij')
        xlabel_text = '$v_{\perp}$ [km/s]'
        ax.set_aspect('equal')
    else:
        print('argument xaxis should be "mu" or "vperp"')
        return
    
    if vmin is None:
        vmin = np.min(Z)
    if vmax is None:
        vmax = np.max(Z)

    if log:
        import matplotlib.colors as mcolors
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        pcm = ax.pcolormesh(X, Y, Z, norm=norm, cmap=cmap, alpha=alpha)
        label = 'log$_{10}$('+label+')'
    else:
        pcm = ax.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha)

    if colorbar:
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label(label)

    if xlabel:
        ax.set_xlabel(xlabel_text, fontsize=18)
    if ylabel:
        ax.set_ylabel('$v_{||}$ [km/s]', fontsize=18)
    if title is not None:
        ax.set_title(title, fontsize=18)

    if not savefile is None:
        fig.savefig(savefile)
        plt.close(fig)
    else:
        if was_fig_none:
            plt.show()

    if return_pcm:
        return pcm


if __name__ == '__main__':
    from chunk_reader import dist_reader, field_reader
    from growthrate import calc_bounce_and_drift_freqs

    i1, i2, i3 = 32, 4, 0

    run = Run('../../run/case2b128')
    run.read('bg')
    run.read('coord')
    trange_v = (24, 25, 1)
    run.set_trange(trange_v, 'v')

    d1, d2, d3, l1, l2, l3 = run.resolve_global_idx(i1, i2, i3)

    file_path_dist = os.path.join(run.prefix, f'dist1-{d1:02d}-{d2:02d}-{d3:02d}.dat')
    dist = dist_reader(file_path_dist, run.N1_local, run.N2_local, run.N3_local, run.Nm, run.Nv, trange_v)
    dist = dist[l3, l2, l1, :, :, :] # (Nm, Nv, Nt)
    file_path_field = os.path.join(run.prefix, f'field-{d1:02d}-{d2:02d}-{d3:02d}.dat')
    V, B = field_reader(file_path_field, run.N1_local, run.N2_local, run.N3_local, run.trange)
    B = B[l3, l2, l1, :, :] # (3, Nt)
    B *= run.unitB # to [nT]
    Babs = np.zeros(run.Nt_v)
    for it in range(run.Nt_v):
        Babs[it] = np.linalg.norm(B[:, it] + run.B0[i3, i2, i1, :]) # [nT]

    fig, axes = plt.subplots(1, 2, figsize=(16,8))

    wb, wd = calc_bounce_and_drift_freqs(run, Babs[it]/run.unitB, i2)
    vperp = convert_mu_to_vperp(run.mu, Babs[it])*run.unitV # [km/s]
    x, y = np.meshgrid(vperp, run.vp*1e-3, indexing='ij')
    draw_on_velocity_space(run, dist[:, :, it], xaxis='vperp', B=Babs[it],
                            fig=fig, ax=axes[0], log=False,
                            cmap='viridis', label='f [s^3/m^6]', title=f'{run.time_v[it]:.1f} s')
    ctr = axes[0].contour(x, y, wd*1e3/(2*np.pi), colors='gray', linestyles='dashed')
    axes[0].clabel(ctr, fmt='%.2f', colors='gray', fontsize=8)  

    print(vperp[-2], run.vp[16]*1e-3, 1/2 * run.Mp * ((vperp[-2]*1e3)**2 + (run.vp[16])**2) / run.Qp/ 1e3)
    plt.show()
