import sys
sys.path.append('..')

import numpy as np

from base import Run


def calc_total_energy(run):
    """
    arguments: run -- Run object
    return   : time series of total energy in the simulation domain [J]
    
    (kinetic energy density [J/m^3]) = 3/2 (total pressure [Pa])
                                     = 1/2 (parallel pressure [Pa]) + (perpendicular pressure [Pa])
    (total energy [J]) = ∫ (kinetic energy density) dV
    """
    energy_density = (0.5 * run.Ppa + run.Ppe) / run.unitP # [J/m^3]
    dV = run.h1 * run.dx1 * run.h2 * run.dx2 * run.h3 * run.dx3 # [m^3]
    total_energy = np.zeros(run.Nt)
    for it in range(run.Nt):
        total_energy[it] = np.sum(energy_density[..., it] * dV) # [J]
    return total_energy
 
def calc_SYM_H(run):
    """
    arguments: run -- Run object
    return   : time series of SYM-H index [nT]

    (SYM-H [nT]) = 4/3 * -2 (total energy [J]) / Me (dipole moment [A m^2])
    4/3 : factor to convert magnetic depression at center of Earth to that on ground
    """
    total_energy = calc_total_energy(run) # [J]
    Me_Am2 = np.abs(run.Me) * 1.0e7 # [T m^3] to [A m^2]
    sym_h = 4/3 * -2 * total_energy / Me_Am2 # [T]
    return sym_h * run.unitB # [nT]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    run = Run('../../run/case1b256')
    run.set_trange((0, 2161, 20))
    run.read('coord')
    run.read('moment')

    total_energy = calc_total_energy(run)
    sym_h = calc_SYM_H(run)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    axes[0].plot(run.time, total_energy)
    axes[0].set_ylabel('Total Energy [J]')
    axes[1].plot(run.time, sym_h)
    axes[1].set_ylabel('SYM-H [nT]')
    axes[1].set_xlabel('Time [s]')
    plt.show()
