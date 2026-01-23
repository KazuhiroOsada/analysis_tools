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
