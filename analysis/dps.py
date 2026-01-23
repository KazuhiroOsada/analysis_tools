import sys
sys.path.append('..')

import numpy as np

from base import Run


def calc_total_energy(run):
    """
    Integrate the kinetic energy density over the simulation domain to get total energy
    (kinetic energy density [J/m^3]) = 3/2 (total pressure [Pa])
                                     = 1/2 (parallel pressure [Pa]) + (perpendicular pressure [Pa])
    (total energy [J]) = \int (kinetic energy density) dV

    Parameters
    ----------
    run : Run object

    Returns
    -------
    total_energy : [J] (Nt,)
    """
    energy_density = (0.5 * run.Ppa + run.Ppe) / run.unitP # [J/m^3]
    dV = run.h1 * run.dx1 * run.h2 * run.dx2 * run.h3 * run.dx3 # [m^3]
    total_energy = np.zeros(run.Nt)
    for it in range(run.Nt):
        total_energy[it] = np.sum(energy_density[..., it] * dV) # [J]
    return total_energy
 
def calc_SYM_H(run):
    """
    Estimate SYM-H index based on DPS relation
    (SYM-H [nT]) = 4/3 * -2 (total energy [J]) / Me (dipole moment [A m^2])
    where 4/3 is a factor to convert magnetic depression at center of Earth to that on ground

    Parameters
    ----------
    run : Run object

    Returns
    -------
    sym_h : [nT], (Nt,)
    """
    total_energy = calc_total_energy(run) # [J]
    Me_Am2 = np.abs(run.Me) * 1.0e7 # [T m^3] to [A m^2]
    sym_h = 4/3 * -2 * total_energy / Me_Am2 # [T]
    return sym_h * run.unitB # [nT]
