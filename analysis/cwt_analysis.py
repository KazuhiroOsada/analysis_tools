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
    filename : .npz file created by compute_cwt.py
    dt       : time step [s] (default: 5.0 s)
    """
    def __init__(self, filename, dt=5.0):
        data = np.load(filename)
        self.freq = data['freq']
        self.cwts = data['cwts']
        self.N3, self.N2, self.n_freq, self.Nt = self.cwts.shape
        self.time = np.arange(self.Nt) * dt
        self.power = np.abs(self.cwts)**2
        # analysis parameters for peak detection and m number estimation
        self.height = None          # absolute height for peak detection
        self.relative_height = -1   # relative height from the maximum log10(power) for peak detection
        self.n = 2                  # half-width of the neighborhood for m number estimation
        self.freq_low_lim = -np.inf # lower frequency limit for peak detection

    def find_max_power_freq_at(self, i3, i2):
        """
        Parameters
        ----------
        i3, i2 : indices in x3 and x2 directions

        Returns
        -------
        powers    : (Nt,) max power at each time
        peakfreqs : (Nt,) frequency at max power
        """
        powers, peakfreq = np.full(self.Nt, np.nan), np.full(self.Nt, np.nan)
        for it in range(self.Nt):
            if self.height is None:
                height = np.max(np.log10(self.power[i3,i2,:,it])) + self.relative_height
            else:
                height = max(np.max(np.log10(self.power[i3,i2,:,it])) + self.relative_height, self.height)
            peaks, _ = find_peaks(np.log10(self.power[i3,i2,:,it]), height=height, )
            if len(peaks) > 0:
                if self.freq[peaks[0]] > self.freq_low_lim:
                    powers[it] = self.power[i3, i2, peaks[0], it]
                    peakfreq[it] = self.freq[peaks[0]]
        return powers, peakfreq

    def find_max_power_freq_all(self):
        self.max_power = np.zeros((self.N3, self.N2, self.Nt))
        self.max_power_freq = np.zeros((self.N3, self.N2, self.Nt))
        for i3 in range(self.N3):
            for i2 in range(self.N2):
                powers, peakfreq = self.find_max_power_freq_at(i3, i2)
                self.max_power[i3, i2, :] = powers
                self.max_power_freq[i3, i2, :] = peakfreq

    def estimate_mnumber_at(self, i3, i2):
        """
        From phases of CWT coefficients at neighboring points, estimate m numbers at (i3, i2)

        Parameters
        ----------
        i3, i2 : indices in x3 and x2 directions
        """
        mnumbers = np.full(self.Nt, np.nan)

        if not hasattr(self, 'max_power_freq'):
            _, peakfreq = self.find_max_power_freq_at(i3, i2)
        else:
            peakfreq = self.max_power_freq[i3, i2, :]

        i3_neighbors = np.arange(i3-self.n, i3+self.n+1) % self.N3
        for it in range(self.Nt):
            if np.isnan(peakfreq[it]):
                continue
            freq_idx = np.where(self.freq == peakfreq[it])[0][0]
            phase_neighbors = np.angle(self.cwts[i3_neighbors, i2, freq_idx, it])
            phase_neighbors = np.unwrap(phase_neighbors)
            dphi = 2*np.pi/self.N3
            phi = np.arange(-self.n,self.n+1)*dphi
            mnumbers[it], _ = np.polyfit(phi, phase_neighbors, 1)
            mnumbers[it] *= -1 # due to the definition of mother wavelet
        return mnumbers

    def estimate_mnumber_all(self):
        self.mnumbers = np.zeros((self.N3, self.N2, self.Nt))
        for i3 in range(self.N3):
            for i2 in range(self.N2):
                self.mnumbers[i3, i2, :] = self.estimate_mnumber_at(i3, i2)

    def estimate_mnumber_all_freq(self, i3, i2):
        """
        Draw m number for each CWT coefficient at (i3, i2)

        Parameters
        ----------
        i3, i2 : indices in x3 and x2 directions
        """
        mnumbers = np.full((self.n_freq, self.Nt), np.nan)
        i3_neighbors = np.arange(i3-self.n, i3+self.n+1) % self.N3
        for ifreq in range(self.n_freq):
            for it in range(self.Nt):
                phase_neighbors = np.angle(self.cwts[i3_neighbors, i2, ifreq, it])
                phase_neighbors = np.unwrap(phase_neighbors)
                dphi = 2*np.pi/self.N3
                phi = np.arange(-self.n,self.n+1)*dphi
                mnumbers[ifreq, it], _ = np.polyfit(phi, phase_neighbors, 1)
                mnumbers[ifreq, it] *= -1 # due to the definition of mother wavelet

        dt = self.time[1] - self.time[0]
        x = np.linspace(self.time[0] - dt/2, self.time[-1] + dt/2, self.Nt+1)
        y = np.zeros(self.n_freq+1)
        y[0] = (3*self.freq[0] - self.freq[1]) / 2
        y[1:-1] = (self.freq[:-1] + self.freq[1:]) / 2
        y[-1] = (3*self.freq[-1] - self.freq[-2]) / 2

        fig, axes = plt.subplots(3,1, figsize=(10,8), sharex=True)
        pcm = axes[0].pcolormesh(x, y, np.log10(self.power[i3,i2,:,:]), cmap='jet')
        fig.colorbar(pcm, ax=axes[0], label='log10(Power)')
        axes[0].set_ylabel('Frequency [Hz]')
        pcm = axes[1].pcolormesh(x, y, mnumbers, cmap='coolwarm', vmin=-20, vmax=20)
        fig.colorbar(pcm, ax=axes[1], label='m number')
        axes[1].set_ylabel('Frequency [Hz]')
        pcm = axes[2].pcolormesh(x, y, np.abs(mnumbers / (self.freq[:, np.newaxis]*1e3)), cmap='plasma', vmin=0, vmax=15)
        fig.colorbar(pcm, ax=axes[2], label='|m/f| [s]')
        axes[2].set_ylabel('Frequency [Hz]')
        axes[2].set_xlabel('Time [s]')
        for ax in axes:
            ax.contour(self.time, self.freq, np.log10(self.power[i3,i2,:,:]), levels=np.arange(-20,0,1),colors='white', linewidths=1)
            ax.set_yscale('log')
        plt.show()
        plt.close()

    def plot_estimation_validity(self, run, i3, i2, low_cutoff=1e-3, high_cutoff=22e-3, t_from=1000, t_to=4000):
        """
        Make a summary plot to check the validity of the analysis at (i3, i2), comparing CWT-based and peak-based m number estimation.

        Parameters
        ----------
        run                     : Run object, with electric field data calculated
        i3, i2                  : indices in x2 and x3 directions
        low_cutoff, high_cutoff : bandpass frequency range [Hz] for peak-based m number estimation
        t_from, t_to            : time range [s] for peak-based m number estimation
        """
        from wavelet import bandpass_filter, draw_power_spectrum

        if not hasattr(self, 'max_power_freq'):
            _, peakfreq = self.find_max_power_freq_at(i3, i2)
        else:
            peakfreq = self.max_power_freq[i3, i2, :]
        mnumbers_cwt = self.estimate_mnumber_at(i3, i2)
        time_peaks, mnumbers_peaks = estimate_mnumber_from_peaks(self, run, i3, i2, n=self.n, low_cutoff=low_cutoff, high_cutoff=high_cutoff, t_from=t_from, t_to=t_to)
        fig, axes = plt.subplots(3, 1, figsize=(10,8), sharex=True)

        axes[0].set_title(f'i3={i3}, i2={i2}, height={self.height}, relative_height={self.relative_height}, n={self.n}')

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
        plt.close()

def estimate_mnumber_from_peaks(cwt_data, run, i3, i2, n=2, t_from=1000, t_to=4000, low_cutoff=1e-3, high_cutoff=22e-3):
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
    Ephi_neighbors = np.array(Ephi_neighbors) # (2n+1, Nt)

    # find peaks in time range at neighboring points
    peak_neighbors = {} # key: -n ... n
    for i, Ephi_n in enumerate(Ephi_neighbors):
        peaks_maximum, _ = find_peaks(Ephi_n)
        peaks_minimum, _ = find_peaks(-Ephi_n)
        peaks = np.sort(np.concatenate((peaks_maximum, peaks_minimum)))
        peaks = peaks[(t_from <= run.time[peaks]) & (run.time[peaks] <= t_to)]
        peak_neighbors[i-n] = peaks

    # frequency estimation by CWT
    _, freq_cwt = cwt_data.find_max_power_freq_at(i3, i2) # (Nt,)

    # estimate m numbers for each peak at (i3, i2)
    time = run.time[peak_neighbors[0]]
    mnum = np.full(len(time), np.nan)
    for ip, peaks in enumerate(peak_neighbors[0]):
        tp = time[ip]
        if np.isnan(freq_cwt[peaks]):
            continue
        for n_try in range(n,0,-1):
            skip_n_try = False
            i3_offsets = np.arange(-n_try, n_try+1)
            tp_neighbors = np.full(len(i3_offsets), np.nan)
            # find corresponding peaks at neighboring points
            for i in i3_offsets:
                t_peaks_i = run.time[peak_neighbors[i]] # (Number of peaks at i3+i)
                min_dt_i = np.min(np.abs(t_peaks_i - tp))
                if min_dt_i > (1/freq_cwt[peaks])/4 * np.abs(i): # neighboring points are supposed to be in 1/4 wavelength
                    skip_n_try = True
                tp_neighbors[i + n_try] = t_peaks_i[np.argmin(np.abs(t_peaks_i - tp))]
            if skip_n_try:
                continue
            a, _ = np.polyfit(i3_offsets, tp_neighbors, 1)
            fit_values = a*i3_offsets + tp
            corr = np.corrcoef(tp_neighbors, fit_values)[0,1] # r was almost 1 in practice
            if corr > 0.9:
                mnum[ip] = a * freq_cwt[peaks] * run.N3 # m = slope * f * N3
                break
    return time, mnum

def make_plot_at_each_point(rundir, cwt_file):
    """
    Plot analysis results at specified points interactively to check validity and set parameters
    """
    run = Run(rundir)
    run.set_trange((0, 1441, 1))
    run.read_equatorial('bg')
    run.read_equatorial('field')
    run.calc_electric_field()

    cwt_data = CwtData(cwt_file)
    cwt_data.height = -6
    cwt_data.relative_height = -3
    cwt_data.freq_low_lim = 1.6e-3

    while True:
        i3, i2 = list(map(int, input('i3, i2 = ').split()))
        cwt_data.plot_estimation_validity(run, i3, i2, low_cutoff=1.6e-3, high_cutoff=6.6e-3, t_from=1500, t_to=3500)

def cwt_analysis_all(cwt_files, prefix='cwt'):
    import os
    import time

    for cwt_file in cwt_files:
        t0 = time.time()
        print(f'Processing {os.path.join(prefix, cwt_file)}...')
        data = CwtData(os.path.join(prefix, cwt_file))
        # set analysis parameters
        data.height = -6
        data.relative_height = -3
        data.freq_low_lim = 1.6e-3

        data.find_max_power_freq_all()
        data.estimate_mnumber_all()
        np.savez(os.path.join(prefix, f'analysis_{cwt_file.replace("/", "_")}'),
                 max_power_freq = data.max_power_freq,
                 max_power      = data.max_power,
                 mnumbers       = data.mnumbers,
                 time           = data.time,
                 height         = data.height,
                 relative_height= data.relative_height,
                 n              = data.n,
                 freq_low_lim   = data.freq_low_lim)
        t1 = time.time()
        print(f'Done in {t1 - t0:.1f} s') # the processing takes ~4min at spacest


if __name__ == '__main__':
    #main2()

    rundir = '../../run/case1b128'
    cwt_file = 'cwt/case1b128_Ephi.npz'
    main3(rundir, cwt_file)

