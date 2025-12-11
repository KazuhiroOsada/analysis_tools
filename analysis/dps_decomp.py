import sys
sys.path.append('..')

import numpy as np

from base import Run
from compute_mom import get_delv, compute_P_vectorized, compute_den_vectorized


def calc_decomposed_energy(run, factor=1.0):
    """
    Integrate the kinetic energy density for each mu

    Parameters
    ----------
    run     : Run object
    factor  : correction factor for sparsed Pressure integration (default: 1.0)

    Returns
    -------
    energy_mu : [J], (Nm, Nt)
    """
    energy_density_mu = (0.5 * run.Ppa_mu + run.Ppe_mu) / run.unitP
    h1 = run.h1[run.x3slice, run.x2slice, run.x1slice, None]
    h2 = run.h2[run.x3slice, run.x2slice, run.x1slice, None]
    h3 = run.h3[run.x3slice, run.x2slice, run.x1slice, None]
    dV = h1 * run.dx1 * h2 * run.dx2 * h3 * run.dx3
    energy_mu = np.zeros((run.Nm, run.Nt_v))
    for it in range(run.Nt_v):
        energy_mu[:, it] = factor * np.sum(energy_density_mu[..., it] * dV, axis=(0,1,2))
    return energy_mu

def calc_decomposed_number(run, factor=1.0):
    """
    Integrate the density for each mu

    Parameters
    ----------
    run     : Run object
    factor  : correction factor for sparsed Density integration (default: 1.0)

    Returns
    -------
    total_number_mu : (Nm, Nt)
    """
    h1 = run.h1[run.x3slice, run.x2slice, run.x1slice, None]
    h2 = run.h2[run.x3slice, run.x2slice, run.x1slice, None]
    h3 = run.h3[run.x3slice, run.x2slice, run.x1slice, None]
    dV = h1 * run.dx1 * h2 * run.dx2 * h3 * run.dx3
    total_number_mu = np.zeros((run.Nm, run.Nt_v))
    for it in range(run.Nt_v):
        total_number_mu[:, it] = factor * np.sum(run.Rho_mu[..., it]/run.unitN * dV, axis=(0,1,2))
    return total_number_mu

def calc_decomposed_SYM_H(run, factor=1.0):
    """
    Estimate SYM-H index for each mu based on DPS relation

    Parameters
    ----------
    run     : Run object
    factor  : correction factor for sparsed Pressure integration (default: 1.0)

    Returns
    -------
    sym_h_mu : [nT], (Nm, Nt)
    """
    energy_mu = calc_decomposed_energy(run, factor=factor)
    Me_Am2 = np.abs(run.Me) * 1.0e7
    sym_h_mu = 4/3 * -2 * energy_mu / Me_Am2
    return sym_h_mu * run.unitB


if __name__ == '__main__':
    from time import time
    import matplotlib.pyplot as plt

    t0 = time()
    run = Run('../../run/case1b128')
    run.read('coord')
    run.read('bg')
    run.set_trange((10, 51, 10), 'v')
    run.read('dist')
    run.read('field')
    run.read('moment')
    run.calc_magnetic_amplitude()
    get_delv(run)
    t1 = time()
    print(f'Data read time: {t1 - t0:.2f} s')

    x1slice = slice(0, run.N1//2, 1)
    x2slice = slice(None, None, None)
    x3slice = slice(None, None, 4)
    factor = 2.0 * 4.0

    t0 = time()
    Pperp, Ppara = compute_P_vectorized(run, x3slice=x3slice, x2slice=x2slice, x1slice=x1slice)
    Rho = compute_den_vectorized(run, x3slice=x3slice, x2slice=x2slice, x1slice=x1slice)
    t1 = time()
    print(f'Computation time: {t1 - t0:.2f} s')

    run.x1slice = x1slice
    run.x2slice = x2slice
    run.x3slice = x3slice
    run.Ppe_mu = Pperp
    run.Ppa_mu = Ppara
    run.Rho_mu  = Rho

    energy_mu = calc_decomposed_energy(run, factor=factor)
    sym_h_mu = calc_decomposed_SYM_H(run, factor=factor)
    total_number_mu = calc_decomposed_number(run, factor=factor)

    from dps import calc_SYM_H

    run.Nt = run.Nt_v
    sym_h = calc_SYM_H(run)
    for it in range(run.Nt_v):
        print(f'Time {run.time_v[it]:.1f} s: SYM-H from total pressure = {sym_h[it]:.2f} nT, SYM-H from decomposed pressure = {np.sum(sym_h_mu[:, it]):.2f} nT')

    y_mu = np.zeros(run.Nm+1)
    log_dm = np.log10(run.mu)[1] - np.log10(run.mu)[0]
    y_mu[1:-1] = 0.5 * (np.log10(run.mu[1:]) + np.log10(run.mu[:-1]))
    y_mu[0] = np.log10(run.mu[0]) - 0.5 * log_dm
    y_mu[-1] = np.log10(run.mu[-1]) + 0.5 * log_dm
    x_t = np.zeros(run.Nt_v+1)
    dt = run.time_v[1] - run.time_v[0]
    x_t[1:-1] = 0.5 * (run.time_v[1:] + run.time_v[:-1])
    x_t[0] = run.time_v[0] - 0.5 * dt
    x_t[-1] = run.time_v[-1] + 0.5 * dt

    X_t, Y_mu = np.meshgrid(x_t, y_mu, indexing='ij')

    plt.pcolormesh(X_t, Y_mu, energy_mu.T)
    plt.show()
