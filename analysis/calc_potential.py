import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from base import Run


def calc_potential(run, E):
    """
    Parameters
    ----------
    run : Run object
    E   : (N3, N2, 3) array of electric field [mV/m] in dipole coordinate

    Returns
    -------
    potential : (N3, N2) array of electric potential [kV]

    Note
    ----
    (i2, i3) = (0, 0) point is set to zero potential
    """
    potential = np.zeros((run.N3, run.N2))
    r = 1 / run.x2**2 # [Re]
    # integrate potential along x2 at i3=0
    for i2 in range(1, run.N2):
        dr = r[i2] - r[i2-1]
        potential[0, i2] = potential[0, i2-1] + 0.5 * (E[0, i2, 1] + E[0, i2-1, 1]) * dr
    # integrate potential along x3
    for i3 in range(1, run.N3):
        potential[i3, :] = potential[i3-1, :] - 0.5 * (r * (E[i3, :, 2] + E[i3-1, :, 2])) * run.dx3
    # periodic condition in x3
    potential_N3 = potential[run.N3-1, :] - 0.5 * (r * (E[0, :, 2] + E[run.N3-1, :, 2])) * run.dx3
    delta = potential_N3 - potential[0, :]
    correction = np.linspace(0, -delta, run.N3)
    potential += correction
    potential *= 1/run.unitE * 1e-3 # kV
    return potential

def calc_Alfven_layer(run, potential, im, it=0):
    """
    Parameters
    ----------
    run       : Run object
    potential : (N3, N2) array of electric potential [kV]
    im        : index of mu axis
    it        : time index (default: 0)

    Return
    ------
    potential : (N3, N2) array of Alfven layer potential [kV]
    """
    return potential + run.mu[im] * run.Babs[..., it] / run.unitB * 1e-3 # kV

if __name__ == '__main__':
    import os
    from draw import draw_equatorial

    rundirs = ['case2b128', 'case3b128', 'case2b128n', 'case3b128n']
    runs = []
    
    for rundir in rundirs:
        run = Run(f'../../run/{rundir}')
        run.read_equatorial('bg')
        run.read('coord')
        run.set_trange((0, 531, 50))
        run.read_equatorial('field')
        run.read('moment')
        run.calc_electric_field()
        run.calc_magnetic_amplitude()
        runs.append(run)

    """
    for it in range(run.Nt):
        im = -9
        fig, axes = plt.subplots(2, 4, figsize=(15, 6))
        fig.suptitle(f'Time = {runs[0].time[it]:.1f} s', fontsize=24)
        for i, run in enumerate(runs):
            axes[0, i].set_title(f'Case {i+1}', fontsize=22)
            potential = calc_potential(run, run.E[..., it])
            alfven_potential = calc_Alfven_layer(run, potential, im=im, it=it)
            ctr = axes[0, i].contour(run.Xi[..., run.N1//2], run.Yi[..., run.N1//2], alfven_potential, colors='white', linewidths=1.0, levels=np.arange(-100, 100, 10))
            axes[0, i].contour(run.Xi[..., run.N1//2], run.Yi[..., run.N1//2], alfven_potential, colors='white', linewidths=0.3, levels=np.arange(-100, 100, 2))
            axes[0, i].clabel(ctr, fmt='%d', colors='white', fontsize=8)
            ctr = axes[1, i].contour(run.Xi[..., run.N1//2], run.Yi[..., run.N1//2], potential, colors='white', linewidths=1.0, levels=np.arange(-100, 100, 10))
            axes[1, i].contour(run.Xi[..., run.N1//2], run.Yi[..., run.N1//2], potential, colors='white', linewidths=0.3, levels=np.arange(-100, 100, 2))
            axes[1, i].clabel(ctr, fmt='%d', colors='white', fontsize=8)
            draw_equatorial(run, run.Ppe[..., run.N1//2, it]+1e-12, fig=fig, ax=axes[0, i], vmin=1e-4, vmax=10, log=True, clabel='$P_\perp$ [nPa]', width=8.0, cfs=20)
            draw_equatorial(run, run.Rho[..., run.N1//2, it], fig=fig, ax=axes[1, i], vmin=1, vmax=1000, log=True, clabel='n [/cc]', width=8.0, cfs=20)
        plt.show()
        plt.close()
    """