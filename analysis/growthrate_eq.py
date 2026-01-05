import os
import sys
sys.path.append('..')
from time import time

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from growthrate import calc_gamma_velocity_space, calc_vperp


def load_cwt_analysis(run, cwt_analysis_file):
    """
    Load cwt analysis data and store in run object
    """
    data = np.load(cwt_analysis_file)
    run.mnumbers = data['mnumbers']
    run.max_freq = data['max_power_freq']
    
def compute_equatorial_growth_rate(run, savefile=None, n=0):
    """
    Parameters
    ----------
    run               : Run object with equatorial 'dist', 'field', 'bg', 'coord' data read
    cwt_analysis_file : path to cwt analysis file
    savefile          : path to save growth rate data
    n                 : harmonic number (default: 0)
    """
    it_cwt = np.rint(run.time_v / (run.ifdiag*run.delt)).astype(int) # for cwt analysis data
    run.gamma  = np.full((run.N3, run.N2, run.Nt_v), np.nan)
    run.gamma1 = np.full((run.N3, run.N2, run.Nt_v), np.nan)
    run.gamma2 = np.full((run.N3, run.N2, run.Nt_v), np.nan)
    run.vperp  = np.full((run.N3, run.N2, run.Nt_v), np.nan)
    run.vpara  = np.full((run.N3, run.N2, run.Nt_v), np.nan)
    run.resf   = np.full((run.N3, run.N2, run.Nt_v), np.nan)
    for i3 in range(run.N3):
        for i2 in range(1, 20): # in range(1, run.N2-1):
            for it_v in range(run.Nt_v):
                dist_m = run.dist[i3, i2-1, :, :, it_v]
                dist   = run.dist[i3, i2  , :, :, it_v]
                dist_p = run.dist[i3, i2+1, :, :, it_v]
                B_m    = run.Babs[i3, i2-1, it_v] / run.unitB
                B      = run.Babs[i3, i2  , it_v] / run.unitB
                B_p    = run.Babs[i3, i2+1, it_v] / run.unitB
                omega = run.max_freq[i3, i2, it_cwt[it_v]] * 2*np.pi # rad/s
                m     = run.mnumbers[i3, i2, it_cwt[it_v]]

                g1_vs, g2_vs, g_vs, _ = calc_gamma_velocity_space(run, dist_m, dist, dist_p, B_m, B, B_p, i2, omega, m, n)

                try:
                    im, iv = np.unravel_index(np.nanargmax(g_vs), g_vs.shape)
                    run.gamma[i3, i2, it_v] = g_vs[im, iv]
                    run.gamma1[i3, i2, it_v] = g1_vs[im, iv]
                    run.gamma2[i3, i2, it_v] = g2_vs[im, iv]
                    run.vperp[i3, i2, it_v] = calc_vperp(run, B)[im] * run.unitV
                    run.vpara[i3, i2, it_v] = run.vp[iv] * run.unitV
                    run.resf[i3, i2, it_v] = dist[im, iv]
                except:
                    pass
    
    if savefile is not None:
        np.savez_compressed(savefile,
                            gamma=run.gamma,
                            gamma1=run.gamma1,
                            gamma2=run.gamma2,
                            vperp=run.vperp,
                            vpara=run.vpara,
                            resf=run.resf,
                            )


if __name__ == '__main__':
    from draw import draw_equatorial

    rundirs = ['../../run/case1b128',
               '../../run/case2b128',
               '../../run/case3b128']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for irun, rundir in enumerate(rundirs):
        run = Run(rundir)
        run.set_trange((0, 25, 24), 'v')
        t0 = time()
        run.read_equatorial('bg')
        run.read_equatorial('coord')
        run.read_equatorial('field')
        run.read_equatorial('dist')
        run.calc_magnetic_amplitude()
        t1 = time()
        print(f'Run {irun}: Data read time: {t1 - t0:.2f} s')
        print(run.time_v)

        cwt_analysis_file = f'cwt/analysis_case{irun+1}b128_Ephi.npz'
        load_cwt_analysis(run, cwt_analysis_file)

        savefile = f'gamma_eq_case{irun+1}b128.npz'
        t0 = time()
        compute_equatorial_growth_rate(run, savefile=savefile, n=0)
        t1 = time()
        print(f'Run {irun}: Growth rate computation time: {t1 - t0:.2f} s')

        draw_equatorial(run, run.gamma[..., -1],fig=fig, ax=axes[irun], title=f'Case {irun+1}', vmin=0, vmax=4e-3, cmap='jet')
    plt.savefig('gamma_equatorial.pdf')
    plt.show()


