import os
import sys
sys.path.append('..')
from time import time

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from growthrate import calc_gamma_velocity_space, calc_vperp


def moving_average_nan(series, m=11):
    n = len(series)
    result = np.full(n, np.nan)

    start = np.argmax(~np.isnan(series))
    end = n - 1 - np.argmax(~np.isnan(series[::-1]))

    # in case all Nan
    if start > end:
        return result
    
    half = m // 2
    for i in range(start, end + 1):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        result[i] = np.nanmean(series[left:right])
    return result

def compute_growth_rate(rundir, cwt_analysis_file=None, 
                        savefile_raw=None, savefile_ma=None, n=0):
    t0 = time()

    run = Run(rundir)
    run.read_equatorial('bg')
    run.read_equatorial('coord')
    run.set_trange((12, 49, 1), 'v')
    run.read_equatorial('field')
    run.calc_magnetic_amplitude()
    run.read_equatorial('dist')

    t1 = time()
    print(f'Data read time: {t1 - t0:.2f} sec')

    prefix = 'processed'
    if cwt_analysis_file is None:
        cwt_analysis_file = os.path.join(prefix, 'analysis', f'{run.prefix.name}.npz')
    data = np.load(cwt_analysis_file)
    it_cwt = np.rint(run.time_v / (run.ifdiag*run.delt)).astype(int)
    freq = data['max_power_freq']
    mnum = data['mnumbers']

    mask = freq < 2.0e-3
    freq[mask] = np.nan
    mnum[mask] = np.nan

    if savefile_raw is None:
        savefile_raw = os.path.join(prefix, 'growthrate',f'{run.prefix.name}_raw.npz')

    run.gamma  = np.full((run.N3, run.N2, run.Nt_v), np.nan)
    run.gamma1 = np.full((run.N3, run.N2, run.Nt_v), np.nan)
    run.gamma2 = np.full((run.N3, run.N2, run.Nt_v), np.nan)
    run.w_res = np.full((run.N3, run.N2, run.Nt_v), np.nan) # kinetic energy [keV]
    run.a_res  = np.full((run.N3, run.N2, run.Nt_v), np.nan) # pitch angle
    run.psd_res = np.full((run.N3, run.N2, run.Nt_v), np.nan)

    for i3 in range(run.N3):
        for i2 in range(1, run.N2-1):
            for it_v in range(run.Nt_v):
                # skip with NaN frequency
                if np.isnan(freq[i3, i2, it_cwt[it_v]]):
                    continue
        
                dist_m = run.dist[i3, i2-1, :, :, it_v]
                dist   = run.dist[i3, i2  , :, :, it_v]
                dist_p = run.dist[i3, i2+1, :, :, it_v]
                B_m    = run.Babs[i3, i2-1, it_v] / run.unitB # T
                B      = run.Babs[i3, i2  , it_v] / run.unitB # T
                B_p    = run.Babs[i3, i2+1, it_v] / run.unitB # T
                omega = freq[i3, i2, it_cwt[it_v]] * 2*np.pi # rad/s
                m     = mnum[i3, i2, it_cwt[it_v]]
                g1_vs, g2_vs, g_vs, _ = calc_gamma_velocity_space(run, dist_m, dist, dist_p, B_m, B, B_p, i2, omega, m, n)

                # if calculated gamma is all NaN, skip
                if np.isnan(g_vs).all():
                    continue
                
                im, iv = np.unravel_index(np.nanargmax(g_vs), g_vs.shape)
                run.gamma[i3, i2, it_v]  = g_vs[im, iv]
                run.gamma1[i3, i2, it_v] = g1_vs[im, iv]
                run.gamma2[i3, i2, it_v] = g2_vs[im, iv]

                vperp = calc_vperp(run, B)[im] # m/s
                vpara = run.vp[iv] # m/s
                run.w_res[i3, i2, it_v] = 1/2 * run.Mp * (vperp**2 + vpara**2) / run.Qp / 1e3 # keV
                run.a_res[i3, i2, it_v] = np.rad2deg(np.arctan2(vperp, vpara))
                run.psd_res[i3, i2, it_v] = dist[im, iv]
    
    np.savez_compressed(savefile_raw,
                        gamma=run.gamma, gamma1=run.gamma1, gamma2=run.gamma2,
                        w_res=run.w_res, a_res=run.a_res, psd_res=run.psd_res,
                        freq=freq, mnum=mnum, time=run.time_v)
    
    t2 = time()
    print(f'Growth rate computation time: {t2 - t1:.2f} sec')

    if savefile_ma is None:
        savefile_ma = os.path.join(prefix, 'growthrate',f'{run.prefix.name}_ma.npz')

    gamma_ma = np.full((run.N3, run.N2, run.Nt_v), np.nan)

    for i3 in range(run.N3):
        for i2 in range(run.N2):
            gamma_ma[i3, i2, :] = moving_average_nan(run.gamma[i3, i2, :], m=11)

    np.savez_compressed(savefile_ma,
                        gamma_ma=gamma_ma,
                        time=run.time_v)

    t3 = time()
    print(f'Moving average time: {t3 - t2:.2f} sec')


if __name__ == '__main__':
    rundirs = ['../../run/case1b128',
               '../../run/case2b128',
               '../../run/case3b128']
    for rundir in rundirs:
        compute_growth_rate(rundir)
        