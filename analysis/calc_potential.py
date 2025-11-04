import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from base import Run


def calc_potential(run, E):
    """
    arguments: run -- Run object
               E   -- (N3, N2, 3) array of electric field [mV/m]
    return: potential -- (N3, N2) array of electric potential [kV]
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


if __name__ == '__main__':
    import os
    from draw import draw_equatorial

    rundir = input('rundir:')
    run = Run(f'../../run/{rundir}')
    run.read_equatorial('bg')
    run.read('coord')
    run.set_trange((0, 1801, 20))
    run.read_equatorial('field')
    run.read_equatorial('moment')
    run.calc_electric_field()

    for it in range(run.Nt):
        potential = calc_potential(run, run.E[..., it])
        fig, axes = plt.subplots(2, 1, figsize=(8, 10))
        fig.suptitle(f'Time = {run.time[it]:.1f} s')
        for ax in axes:
            ctr = ax.contour(run.Xi[..., run.N1//2], run.Yi[..., run.N1//2], potential, colors='white', linewidths=1.0, levels=np.arange(-40, 41, 10))
            ax.contour(run.Xi[..., run.N1//2], run.Yi[..., run.N1//2], potential, colors='white', linewidths=0.3, levels=np.arange(-40, 41, 2))
            ax.clabel(ctr, fmt='%d', colors='white', fontsize=8)
        draw_equatorial(run, run.Rho[..., it], fig=fig, ax=axes[0], vmin=1, vmax=800, log=True, clabel='Density [/cc]')
        draw_equatorial(run, run.Ppe[..., it], fig=fig, ax=axes[1], vmin=1e-4, vmax=10, log=True, clabel='$P_\perp$ [nPa]')
        if not os.path.exists(f'potential_{rundir}'):
            os.makedirs(f'potential_{rundir}')
        plt.savefig(os.path.join(f'potential_{rundir}', f'{it:04d}.png'))
        plt.close()
