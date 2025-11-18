import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from base import Run


class CwtData:
    """
    Parameters
    ----------
    filename : .npz file containing 'freq' and 'cwts'
    dt       : time step [s] (default: 5.0 s)
    """
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
            mnumbers[it] *= -1 # due to the definition of mother wavelet
        return mnumbers

    def estimate_mnumber_all(self, n=2):
        self.mnumbers = np.zeros((self.N3, self.N2, self.Nt))
        for i3 in range(self.N3):
            for i2 in range(self.N2):
                self.mnumbers[i3, i2, :] = self.estimate_mnumber_at(i3, i2, n=n)
    
    def plot_estimation_validity(self, run, i3, i2, n=2, low_cutoff=1e-3, high_cutoff=22e-3, t_from=1000, t_to=4000):
        """
        Make a summary plot to check the validity of the analysis at (i3, i2), comparing CWT-based and peak-based m number estimation.

        Parameters
        ----------
        run    : Run object, with electric field data calculated
        i3, i2 : indices in x2 and x3 directions
        """
        from wavelet import bandpass_filter, draw_power_spectrum

        _, peakfreq = self.find_max_power_freq_at(i3, i2)
        mnumbers_cwt = self.estimate_mnumber_at(i3, i2, n=n)
        time_peaks, mnumbers_peaks = estimate_mnumber_from_peaks(self, run, i3, i2, n=n, low_cutoff=low_cutoff, high_cutoff=high_cutoff, t_from=t_from, t_to=t_to)
        fig, axes = plt.subplots(3, 1, figsize=(10,8), sharex=True)

        axes[0].set_title(f'i3={i3}, i2={i2}')

        axes[0].plot(run.time, bandpass_filter(run, run.E[(i3-2)%run.N3,i2,2,:], low_cutoff, high_cutoff), label=f'i3={(i3-2)%run.N3}')
        axes[0].plot(run.time, bandpass_filter(run, run.E[i3           ,i2,2,:], low_cutoff, high_cutoff), label=f'i3={i3}')
        axes[0].plot(run.time, bandpass_filter(run, run.E[(i3+2)%run.N3,i2,2,:], low_cutoff, high_cutoff), label=f'i3={(i3+2)%run.N3}')
        axes[0].legend()
        axes[0].set_ylabel('Ephi [mV/m]')

        draw_power_spectrum(run, run.E[i3,i2,2,:], fig=fig, ax=axes[1], fmin=low_cutoff, fmax=high_cutoff, fontsize=20, unit='mV/m', label='$E_\phi$')
        axes[1].plot(run.time, peakfreq, color='white',lw=2)

        axes[2].plot(run.time, mnumbers_cwt, label='CWT-based', color='tab:blue')
        axes[2].scatter(time_peaks, mnumbers_peaks, label='Peak-based', color='tab:orange', s=10)
        axes[2].grid()
        axes[2].set_ylabel('m number')
        axes[2].set_xlabel('Time [s]')
        axes[2].legend()

        plt.show()

def estimate_mnumber_from_peaks(cwt_data, run, i3, i2, t_from=1000, t_to=4000, n=2, low_cutoff=1e-3, high_cutoff=22e-3):
    """
    Estimate m numbers from time of peaks of the poloidal electric field time series at (i3, i2) and its neighboring points.

    Parameters
    ----------
    cwt_data     : CwtData object
    run          : Run object, with electric field data calculated
    i3, i2       : indices of the spatial point
    t_from, t_to : time range [s] for the estimation
    n            : number of neighboring points to consider

    Returns
    -------
    time_peaks : times of detected peaks [s]
    mnum_peaks : estimated m numbers at the peak times

    Note
    ----
    In this function, it is assumed that the wave is almost coherent in the azimuthal direction within the range of consideration,
    therefore, each peak in the time series at (i3, i2) are expected to correspond to peaks at neighboring i3 points.
    Time of these peaks, t, will satisfy the relation:
        t = m/omega * (phi-phi0) + t0
    where phi is the azimuthal angle, omega is the angular frequency, and m is the azimuthal wave number.
    The relation can be rewritten as:
        t = m/(f*N3) * (i3-i30) + t0
    Then m number can be estimated by linear fitting of the peak times at neighboring points.
    """
    from wavelet import bandpass_filter
    from scipy.signal import find_peaks

    i3_neighbors = np.arange(i3-n, i3+n+1) % run.N3

    # bandpass Ephi at neighboring points
    Ephi_neighbors = []
    for i3_n in i3_neighbors:
        Ephi_n = bandpass_filter(run, run.E[i3_n, i2, 2, :], low_cutoff, high_cutoff)
        Ephi_neighbors.append(Ephi_n)

    # find peaks in time range at neighboring points
    peaks_list = []
    for Ephi_n in Ephi_neighbors:
        peaks_maximum, _ = find_peaks(Ephi_n)
        peaks_minimum, _ = find_peaks(-Ephi_n)
        peaks = np.sort(np.concatenate((peaks_maximum, peaks_minimum)))
        peaks = peaks[(t_from <= run.time[peaks]) & (run.time[peaks] <= t_to)]
        peaks_list.append(peaks)

    # frequency estimation by CWT
    _, freq_cwt = cwt_data.find_max_power_freq_at(i3, i2) # (Nt,)

    # estimate m numbers for each peak
    time_peaks = run.time[peaks_list[n]]
    mnum_peaks = np.full(len(time_peaks), np.nan)
    for i, peak in enumerate(peaks_list[n]):
        if np.isnan(freq_cwt[peak]):
            continue
        for m in range(n,0,-1):
            i3_offsets = np.arange(-m, m+1)
            times = [run.time[peaks_list[j][i]] for j in range(n-m, n+m+1)]
            a, _ = np.polyfit(i3_offsets, times, 1)
            fit_values = a*i3_offsets + times[m]
            corr = np.corrcoef(times, fit_values)[0,1] # r was almost 1 in practice
            if corr > 0.9:
                mnum_peaks[i] = a * freq_cwt[peak] * run.N3 # m = slope * f * N3
                break
    return time_peaks, mnum_peaks

def main1():
    run = Run('../../run/case2b256new')
    run.set_trange((0, 2161, 1))
    run.read_equatorial('bg')
    run.read_equatorial('field')
    run.calc_electric_field()

    cwt_data = CwtData('../../cwt/case2/Pc5_Ephi.npz')
    while True:
        i3, i2 = list(map(int, input('i3, i2 = ').split()))
        cwt_data.plot_estimation_validity(run, i3, i2, low_cutoff=1e-3, high_cutoff=7e-3, t_from=1500, t_to=3500)

def main2():
    import os
    import time

    prefix = '../../cwt'
    cwt_files = ['case1/Pc5_Ephi.npz',
                  'case2/Pc5_Ephi.npz',
                  'case3/Pc5_Ephi.npz']

    for cwt_file in cwt_files:
        t0 = time.time()
        print(f'Processing {os.path.join(prefix, cwt_file)}...')
        data = CwtData(os.path.join(prefix, cwt_file))
        data.find_max_power_freq_all()
        data.estimate_mnumber_all(n=2)
        np.savez(f'cwt_analysis_{cwt_file.replace("/", "_")}',
                 max_power_freq = data.max_power_freq,
                 max_power      = data.max_power,
                 mnumbers       = data.mnumbers)
        t1 = time.time()
        print(f'Done in {t1 - t0:.1f} s') # the processing takes ~ 10 minutes

def main3():
    cwt_data = CwtData('../../cwt/case1/Pc5_Ephi.npz')

    i3, i2 = 10, 4
    n = 2

    peakfreq = cwt_data.find_max_power_freq_at(i3, i2)[1]

    i3_neighbors = np.arange(i3-n, i3+n+1) % cwt_data.N3
    for it in range(cwt_data.Nt):
        if np.isnan(peakfreq[it]):
            continue
        freq_idx = np.where(cwt_data.freq == peakfreq[it])[0][0]
        phase_neighbors = np.angle(cwt_data.cwts[i3_neighbors, i2, freq_idx, it])
        phase_neighbors_unwrapped = np.unwrap(phase_neighbors)
        dphi = 2*np.pi/cwt_data.N3
        phi = np.arange(-n,n+1)*dphi
        mnumber, _ = np.polyfit(phi, phase_neighbors_unwrapped, 1)
        fig, ax = plt.subplots()
        ax.plot(phi, phase_neighbors_unwrapped, 'o', label='unwrapped')
        ax.plot(phi, mnumber*phi + phase_neighbors_unwrapped[n], '-', label='fit')
        ax.plot(phi, phase_neighbors, 'x', label='wrapped')
        ax.legend()
        ax.set_title(f't={cwt_data.time[it]:.1f} s, mnumber={mnumber:.2f}')
        plt.show()


if __name__ == '__main__':
    main1()
