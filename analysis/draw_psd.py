import sys
sys.path.append('..')
import os

import numpy as np
import matplotlib.pyplot as plt

from base import Run


# Note : it is assumed that these functions are assumed to be used when single species is simulated

def convert_mu_to_vperp(mu, B):
    """
    arguments: mu -- magnetic moment [eV/T]
               B  -- magnetic field [nT]
    return   : vperp -- perpendicular velocity [m/s]
    """
    Qp = 1.602e-19 # proton charge [C]
    Mp = 1.673e-27 # proton mass [kg]
    return np.sqrt(2 * mu * (B*1e-9) * Qp / Mp)

def draw_on_velocity_space(run, Z, xaxis='mu', B=None, fig=None, ax=None,
                           log=False, vmin=None, vmax=None, cmap='viridis',
                           label='', title='', savefile=None):
    """
    arguments: run -- Run object
               Z   -- (Nm, Nv) array to be drawn
               xaxis -- 'mu' or 'vperp' (default: 'mu')
               B   -- magnetic field [nT] at the grid point where Z is calculated, required when xaxis='vperp'
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    log_dm = (np.log10(run.mu[-1]) - np.log10(run.mu[0])) / (run.Nm - 1)
    x = 10**np.linspace(np.log10(run.mu[0]) - log_dm/2, np.log10(run.mu[-1]) + log_dm/2, run.Nm+1)
    dv = (run.vp[-1] - run.vp[0]) / (run.Nv - 1)
    y = np.linspace(run.vp[0] - dv/2, run.vp[-1] + dv/2, run.Nv+1) * 1e-3 # to km/s

    if xaxis == 'mu':
        x *= 1e-12 # eV/nT to keV/nT
        X, Y = np.meshgrid(x,y)
        ax.set_xscale('log')
        ax.set_xlabel('mu [keV/nT]')
    elif xaxis == 'vperp':
        if B is None:
            print('argument B is required when xaxis="vperp"')
            return
        x = convert_mu_to_vperp(x, B) * 1e-3 # to km/s
        X, Y = np.meshgrid(x,y)
        ax.set_xlabel('$v_{\perp}$ [km/s]')
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
        pcm = ax.pcolormesh(X, Y, Z.T, norm=norm, cmap=cmap)
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label('log$_{10}$('+label+')')
    else:
        pcm = ax.pcolormesh(X, Y, Z.T, vmin=vmin, vmax=vmax, cmap=cmap)
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label(label)

    ax.set_ylabel('$v_{||}$ [km/s]')
    ax.set_title(title)

    if not savefile is None:
        fig.savefig(savefile)
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    from chunk_reader import dist_reader, field_reader

    i1, i2, i3 = 32, 4, 20

    run = Run('../../run/case1b256')
    run.read('bg')
    trange_v = (0, 101, 10)
    run.set_trange(trange_v, 'v')

    d1, l1 = i1 // run.N1_local, i1 % run.N1_local
    d2, l2 = i2 // run.N2_local, i2 % run.N2_local
    d3, l3 = i3 // run.N3_local, i3 % run.N3_local

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

    for it in range(1, run.Nt_v):
        draw_on_velocity_space(run, dist[:, :, it], xaxis='vperp', B=Babs[it],
                               fig=None, ax=None, log=True,
                               cmap='viridis', label='f [s^3/m^6]', title=f'Time = {run.time_v[it]:.1f} s')
