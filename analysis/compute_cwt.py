import os
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from wavelet import wavelet_transform, get_freq


def find_bandpass_indices(freq, low_cutoff, high_cutoff):
    """
    Parameters
    ----------
    freq        : (J+1,) frequencies in reverse order
    low_cutoff  : low cutoff frequency [Hz]
    high_cutoff : high cutoff frequency [Hz]

    Returns
    -------
    i_start, i_end : indices for bandpass range
    """
    mask = (low_cutoff <= freq) & (freq <= high_cutoff)
    idx = np.where(mask)[0]
    i_start = idx[0] if len(idx) > 0 else None
    i_end = idx[-1] if len(idx) > 0 else None
    return i_start, i_end

def compute_cwt_equatorial(run, series, low_cutoff, high_cutoff):
    """
    Parameters
    ----------
    run        : Run object
    series     : (N3, N2, Nt) array of time series data
    low_cutoff : low cutoff frequency [Hz]
    high_cutoff: high cutoff frequency [Hz]

    Returns
    -------
    freq : (n_freq,) frequencies in bandpass range
    cwts : (N3, N2, n_freq, Nt) array of CWT coefficients
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
