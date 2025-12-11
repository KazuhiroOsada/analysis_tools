import os
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from base import Run


def get_delv(run):
    """
    Get grid spacings in velocity space
    """
    run.delv = (run.vp[1] - run.vp[0]) # m/s
    log_dm = np.log10(run.mu)[1] - np.log10(run.mu)[0]
    mu_extend = np.zeros(run.Nm+2)
    mu_extend[1:-1] = run.mu
    mu_extend[0] = 10**(np.log10(run.mu)[0] - log_dm)
    mu_extend[-1] = 10**(np.log10(run.mu)[-1] + log_dm)
    run.delm = 0.5 * (mu_extend[2:] - mu_extend[:-2])

def compute_den_vectorized(run, x3slice=None, x2slice=None, x1slice=None, itslice=None):
    """
    Parameters
    ----------
    run                         : Run object
    x3slice, x2slice, x1slice   : slice for x3, x2, x1 axes (default: None, use all)
    itslice                     : slice for time axis (default: None, use all)

    Returns
    -------
    den : density for each mu [1/m^3], shape = (xshape, Nm)

    Note
    ----
    The computation is so heavy, then it is recommended to use slices to reduce the data size.
    """
    def full_slice(s):
        return s if s is not None else slice(None)
    
    x3slice = full_slice(x3slice)
    x2slice = full_slice(x2slice)
    x1slice = full_slice(x1slice)
    itslice = full_slice(itslice)

    dist = run.dist[x3slice, x2slice, x1slice, :, :, itslice]
    B    = run.Babs[x3slice, x2slice, x1slice, None, None, itslice] / run.unitB
    delm = run.delm[None, None, None, :, None, None]

    den = np.sum( 2*np.pi * run.Qm * B * dist
                * run.delv * delm, axis=4) # (x3, x2, x1, Nm, Nv, Nt) -> (x3, x2, x1, Nm, Nt)
    return den * run.unitN

def compute_P_vectorized(run, x3slice=None, x2slice=None, x1slice=None, itslice=None):
    """
    Parameters
    ----------
    run                         : Run object
    x3slice, x2slice, x1slice   : slice for x3, x2, x1 axes (default: None, use all)
    itslice                     : slice for time axis (default: None, use all)

    Returns
    -------
    Pperp : perpendicular pressure for each mu [Pa], shape = (xshape, Nm, Nt)
    Ppara : parallel pressure for each mu [Pa], shape = (xshape, Nm, Nt)

    Note
    ----
    The computation is so heavy, then it is recommended to use slices to reduce the data size.
    The result will not be the same with diagnosed density, and be slightly (~several %) smaller,
    because in the actual moment calculation B^* is used, and here B is used for simplicity.
    The purpose of this function is to see how much each mu contributes to the total pressure, then this underestimation is not serious.
    """
    def full_slice(s):
        return s if s is not None else slice(None)
    
    x3slice = full_slice(x3slice)
    x2slice = full_slice(x2slice)
    x1slice = full_slice(x1slice)
    itslice = full_slice(itslice)

    dist = run.dist[x3slice, x2slice, x1slice, :, :, itslice]
    B    = run.Babs[x3slice, x2slice, x1slice, None, None, itslice] / run.unitB
    mu   = run.mu[None, None, None, :, None, None]
    vp   = run.vp[None, None, None, None, :, None]
    delm = run.delm[None, None, None, :, None, None]

    Pperp = np.sum( 2*np.pi * run.Qm * B * dist
                  * mu * run.Qp * B
                  * run.delv * delm, axis=4) # (x3, x2, x1, Nm, Nv, Nt) -> (x3, x2, x1, Nm, Nt)
    Ppara = np.sum( 2*np.pi * run.Qm * B * dist
                  * run.Mp * vp**2
                  * run.delv * delm, axis=4) # (x3, x2, x1, Nm, Nv, Nt) -> (x3, x2, x1, Nm, Nt)
    return Pperp * run.unitP, Ppara * run.unitP


if __name__ == '__main__':
    from time import time

    run = Run('../../run/case1b128')
    run.read('bg')
    run.set_trange((10, 51, 10), 'v')
    run.read('dist')
    run.read('field')
    run.calc_magnetic_amplitude()
    get_delv(run)

    x1slice = slice(0, run.N1//2, 1)
    x2slice = None
    x3slice = slice(None, None, 2)

    t0 = time()
    Pperp, Ppara = compute_P_vectorized(run, x3slice=x3slice, x2slice=x2slice, x1slice=x1slice)    
    t1 = time()
    print(f'Computation time: {t1 - t0:.2f} s')
