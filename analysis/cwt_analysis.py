import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from base import Run
from draw import draw_equatorial
from wavelet import draw_power_spectrum, bandpass_filter


class cwt_data:
    def __init__(self, filename, dt=5.0):
        """
        arguments: filename -- path to .npz file containing 'freq' and 'cwts'
        """
        data = np.load(filename)
        self.freq = data['freq']
        self.cwts = data['cwts']
        self.N3, self.N2, self.n_freq, self.Nt = self.cwts.shape
        self.time = np.arange(self.Nt) * dt
        self.power = np.abs(self.cwts)**2
    
    def find_max_power_freq_at(self, i3, i2, height=None):
        powers, peakfreq = np.full(self.Nt, np.nan), np.full(self.Nt, np.nan)
        for it in range(self.Nt):
            if height is None:
                height = np.max(np.log10(self.power[i3,i2,:,it])) - 1
            peaks, _ = find_peaks(np.log10(self.power[i3,i2,:,it]), height=height)
            if len(peaks) > 0:
                powers[it] = self.power[i3, i2, peaks[0], it]
                peakfreq[it] = self.freq[peaks[0]]
        return powers, peakfreq

    def find_max_power_freq_all(self, height=None):
        self.max_power = np.zeros((self.N3, self.N2, self.Nt))
        self.max_power_freq = np.zeros((self.N3, self.N2, self.Nt))
        for i3 in range(self.N3):
            for i2 in range(self.N2):
                powers, peakfreq = self.find_max_power_freq_at(i3, i2, height=height)
                self.max_power[i3, i2, :] = powers
                self.max_power_freq[i3, i2, :] = peakfreq

    def estimate_mnumber_at(self, i3, i2, n=2):
        mnumbers = np.full(self.Nt, np.nan)

        if not hasattr(self, 'max_power_freq'):
            _, peakfreq = self.find_max_power_freq_at(i3, i2)
        else:
            peakfreq = self.max_power_freq[i3, i2, :]

        i3_neighbors = np.arange(i3-n, i3+n+1) % self.N3
        for it in range(self.Nt):
            if np.isnan(peakfreq[it]):
                continue
            freq_idx = np.where(self.freq == peakfreq[it])[0][0]
            phase_neighbors = np.angle(self.cwts[i3_neighbors, i2, freq_idx, it])
            phase_neighbors = np.unwrap(phase_neighbors)
            dphi = 2*np.pi/self.N3
            phi = np.arange(-n,n+1)*dphi
            mnumbers[it], _ = np.polyfit(phi, phase_neighbors, 1)
            mnumbers *= -1 # possibly due to the definition of mother wavelet
        return mnumbers

    def estimate_mnumber_all(self, n=2):
        self.mnumbers = np.zeros((self.N3, self.N2, self.Nt))
        for i3 in range(self.N3):
            for i2 in range(self.N2):
                self.mnumbers[i3, i2, :] = self.estimate_mnumber_at(i3, i2, n=n)
    
    def plot_estimation_validity(self, i3, i2):
        """
            while True:
        i3, i2 = list(map(int, input('i3 i2: ').split()))
        powers, peakfreq = data.find_max_power_freq_at(i3, i2, height=None)
        mnumbers = data.estimate_mnumber_at(i3, i2, n=2)
        fig, axes = plt.subplots(2, 1, figsize=(10,6))
        draw_power_spectrum(run, run.E[i3,i2,2,:], fig=fig, ax=axes[0],
                            fmin=1e-3, fmax=22e-3)
        axes[0].plot(run.time, peakfreq, 'w-', lw=2)
        axes[1].plot(run.time, mnumbers)
        axes[1].set_ylabel('Azimuthal wave number m')
        axes[1].set_xlabel('Time [s]')
        ax2 = axes[1].twinx()
        ax2.plot(run.time, np.abs(peakfreq / mnumbers), color='tab:orange')
        ax2.set_ylabel('m / f [Hz]')
        ax2.tick_params(axis='y')
        axes[0].set_xlim(0, 6000)
        axes[1].set_xlim(0, 6000)
        plt.suptitle(f'i3={i3}, i2={i2}')
        plt.show()
        """
        pass


def estimate_mnumber_from_peaks(run, series, i3, i2, n=2):
    """
    arguments: run -- Run object (data must be read at every time step)
               series -- (N3, N2, Nt) array of time series data
               i3, i2 -- indices of the spatial point
    return: mnumbers -- (Nt,) estimated azimuthal wave numbers
    """
    i3_neighbors = np.arange(i3-n, i3+n+1) % run.N3
    return





    

if __name__ == '__main__':
    import os
    import time

    prefix = '../../cwt'
    data_files = ['case1/Pc5_Ephi.npz',
                  'case2/Pc5_Ephi.npz',
                  'case3/Pc5_Ephi.npz']
    # rundirs = ['case1b256', 'case2b256new', 'case3b256']

    for data_file in data_files:
        t0 = time.time()
        print(f'Processing {os.path.join(prefix, data_file)}...')
        data = cwt_data(os.path.join(prefix, data_file))
        data.find_max_power_freq_all()
        data.estimate_mnumber_all(n=2)
        np.savez(f'cwt_analysis_{data_file.replace("/", "_")}',
                 max_power_freq = data.max_power_freq,
                 max_power      = data.max_power,
                 mnumbers       = data.mnumbers)
        t1 = time.time()
        print(f'Done in {t1 - t0:.1f} s') # the processing takes ~ 10 minutes
