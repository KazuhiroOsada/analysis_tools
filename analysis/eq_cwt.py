import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from wavelet import wavelet_transform, get_freq


def find_bandpass_indices(freq, low_cutoff, high_cutoff):
    """
    arguments: freq        -- (J+1,) frequencies in reverse order
               low_cutoff  -- low cutoff frequency [Hz]
               high_cutoff -- high cutoff frequency [Hz]
    return: i_start, i_end -- indices for bandpass range
    """
    mask = (low_cutoff <= freq) & (freq <= high_cutoff)
    idx = np.where(mask)[0]
    i_start = idx[0] if len(idx) > 0 else None
    i_end = idx[-1] if len(idx) > 0 else None
    return i_start, i_end

def compute_cwt_equatorial(run, series, low_cutoff, high_cutoff):
    """
    arguments: run         -- Run object (data must be read at every time step)
               series      -- (N3, N2, Nt) array of time series data
               low_cutoff  -- low cutoff frequency [Hz]
               high_cutoff -- high cutoff frequency [Hz]
    """
    # wavelet transform parameters
    freq = get_freq(run)
    i_start, i_end = find_bandpass_indices(freq, low_cutoff, high_cutoff)
    n_freq = i_end - i_start + 1
    cwts = np.zeros((run.N3, run.N2, n_freq, run.Nt), dtype=complex)
    
    for i3 in range(run.N3):
        for i2 in range(run.N2):
            _, cwt = wavelet_transform(run, series[i3, i2, :])
            cwts[i3, i2, :, :] = cwt[i_start:i_end+1, :]
    return freq[i_start:i_end+1], cwts

def main(rundir, trange, low_cutoff, high_cutoff, cwt_prefix):
    import time # this script will take ~5 min to run

    t0 = time.time()

    run = Run(rundir)
    run.set_trange(trange)
    run.read_equatorial('bg')
    run.read_equatorial('field')
    run.calc_electric_field()

    t1 = time.time()
    print(f'reading data: {t1-t0:.2f} sec')

    # Ephi
    freq, cwts = compute_cwt_equatorial(run, run.E[...,2,:], low_cutoff, high_cutoff)
    np.savez(f'{cwt_prefix}_Ephi.npz', freq=freq, cwts=cwts)
    # Er
    freq, cwts = compute_cwt_equatorial(run, -run.E[...,1,:], low_cutoff, high_cutoff)
    np.savez(f'{cwt_prefix}_Er.npz', freq=freq, cwts=cwts)
    # Bpara
    freq, cwts = compute_cwt_equatorial(run, -run.B[...,0,:], low_cutoff, high_cutoff)
    np.savez(f'{cwt_prefix}_Bpara.npz', freq=freq, cwts=cwts)

    t2 = time.time()
    print(f'wavelet transform: {t2-t1:.2f} sec')


if __name__ == '__main__':
    rundir = '../../run/case1b256'
    trange = (0, 2161, 1)
    low_cutoff = 1.5e-3
    high_cutoff = 7e-3
    cwt_prefix = 'cwt/case1/Pc5'
    main(rundir, trange, low_cutoff, high_cutoff, cwt_prefix)
