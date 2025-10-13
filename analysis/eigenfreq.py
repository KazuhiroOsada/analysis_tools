import sys
sys.path.append('..')
import os

import numpy as np
import matplotlib.pyplot as plt

from base import Run


def calc_Alfven_speed(run, B, Rho):
    """
    arguments: run -- Run object
               B   -- magnetic field [T]
               Rho -- density [/m^3]
    return: vA -- Alfven speed [m/s]
    """
    return np.abs(B) / np.sqrt(run.mu0 * Rho * run.Mp)

def calc_field_line_length(run, i2, i3):
    """
    arguments: run -- Run object
    return: length -- field line length [m]
    """
    return np.sum(run.h1[i3, i2, :] * run.dx1) # m

def calc_eigenfreq(run, i2, i3):
    """
    arguments: run -- Run object
    return: frequencies -- time series of eigenfrequency [/s]
    """
    frequencies = np.zeros(run.Nt)
    for it in range(run.Nt):
        period = 0
        for i1 in range(run.N1):
            B = run.B0[i3, i2, i1, :] + run.B[i3, i2, i1, :, it]
            Babs = np.sqrt(B[..., 0]**2 + B[..., 1]**2 + B[..., 2]**2) / run.unitB # T
            Rho = run.Rho[i3, i2, i1, it] / run.unitN # /m^3
            vA = calc_Alfven_speed(run, Babs, Rho) # m/s
            dl = run.h1[i3, i2, i1] * run.dx1 # m
            period += dl / vA
        period *= 2 # back and forth
        frequencies[it] = 1 / period # Hz
    return frequencies


if __name__ == "__main__":
    i2, i3 = 4, 20
    trange = (0, 2161, 20)
    rundir = '../../run/case1b256'

    run = Run(rundir)
    run.read('coord')
    run.read('bg')
    run.set_trange(trange)
    run.read('field')
    run.read('moment')
    print(f'field line length = {calc_field_line_length(run, i2, i3):.2f} [m]')
    frequencies = calc_eigenfreq(run, i2, i3)
    plt.plot(run.time, frequencies)
    plt.show()
    