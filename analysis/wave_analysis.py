import os
import sys
sys.path.append('..')
import time

import numpy as np

from base import Run
from compute_cwt import compute_cwt_equatorial
from cwt_analysis import CwtData


def wave_analysis(rundir, processed_dir):
    t0 = time.time()

    run = Run(rundir)
    run.read_equatorial('bg')
    run.set_trange((0, 1441, 1))
    run.read_equatorial('field')
    run.calc_electric_field()

    t1 = time.time()
    print(f'Data read time: {t1 - t0:.2f} sec') # ~50 sec for case1b128

    # compute CWT for Ephi component
    cwt_prefix = os.path.join(processed_dir, 'cwt', run.prefix.name)
    low_cutoff = 1.0e-3
    high_cutoff = 7.0e-3
    freq, cwts = compute_cwt_equatorial(run, run.E[..., 2, :], low_cutoff, high_cutoff)
    cwt_path = f'{cwt_prefix}_Ephi.npz'
    np.savez(cwt_path, freq=freq, cwts=cwts, low_cutoff=low_cutoff, high_cutoff=high_cutoff)

    t2 = time.time()
    print(f'Wavelet transform time: {t2 - t1:.2f} sec') # ~30 sec for case1b128

    # analyze CWT data
    cwtdata = CwtData(cwt_path)
    # peak detection criteria
    cwtdata.height = -6
    cwtdata.relative_height = -3
    cwtdata.freq_low_lim = 2.0e-3

    cwtdata.find_max_power_freq_all()
    cwtdata.estimate_mnumber_all()

    analysis_prefix = os.path.join(processed_dir, 'analysis', run.prefix.name)
    analysis_path = f'{analysis_prefix}.npz'
    np.savez(analysis_path,
             max_power_freq = cwtdata.max_power_freq,
             max_power      = cwtdata.max_power,
             mnumbers       = cwtdata.mnumbers,
             time           = cwtdata.time,
             height         = cwtdata.height,
             relative_height= cwtdata.relative_height,
             n              = cwtdata.n,
             freq_low_lim   = cwtdata.freq_low_lim)

    t3 = time.time()
    print(f'Analysis time: {t3 - t2:.2f} sec') # ~300-400 sec for case1b128


if __name__ == '__main__':
    rundirs = ['../../run/case1b128',
               '../../run/case2b128',
               '../../run/case3b128']
    processed_dir = 'processed'

    for rundir in rundirs:
        wave_analysis(rundir, processed_dir)
