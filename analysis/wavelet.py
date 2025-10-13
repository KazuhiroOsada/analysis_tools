import sys
sys.path.append('..')

import numpy as np
from numpy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as patheffects

from base import Run


def bandpass_filter(run, series, low_cutoff, high_cutoff):
    """
    arguments: run         -- Run object (data must be read at every time step)
               series      -- (Nt,) time series data
               low_cutoff  -- low cutoff frequency [Hz]
               high_cutoff -- high cutoff frequency [Hz]
    return: series_filtered -- (Nt,) filtered time series data
    """
    dt = run.delt * run.ifdiag
    window = np.hanning(run.Nt)
    acf = run.Nt / np.sum(window) # amplitude correction factor

    series_ft = fft(series * window)
    freqs = fftfreq(run.Nt, d=dt)
    bandpass = np.zeros(len(freqs))
    bandpass[np.where((np.abs(freqs) >= low_cutoff) & (np.abs(freqs) <= high_cutoff))] = 1.0
    series_ft_filtered = series_ft * bandpass * acf

    series_filtered = ifft(series_ft_filtered).real
    return series_filtered

def wavelet_transform(run, series):
    """
    arguments: run    -- Run object (data must be read at every time step)
               series -- (Nt,) time series data
    return: freq -- (J,) frequencies
            cwt  -- (J+1, Nt) wavelet transform coefficients
    """
    # constants for Morlet wavelet
    n_cwt = int(2**(np.ceil(np.log2(len(series)))))
    dj = 0.125
    omega0 = 6.0
    assert(run.trange[2] == 1), 'data must be read at every time step'
    dt = run.delt * run.ifdiag
    s0 = 2 * dt

    J = int(np.log2(n_cwt * dt / s0) / dj)
    s = s0 * 2**(dj * np.arange(0, J+1, 1))
    omega = 2 * np.pi * np.fft.fftfreq(n_cwt, dt)
    
    series_padded = np.zeros(n_cwt)
    series_padded[:len(series)] = series - np.mean(series) # zero padding and remove mean
    series_ft = fft(series_padded)

    cwt = np.zeros((J+1, n_cwt), dtype=complex)
    Hev = np.array( omega > 0.0, dtype=float ) # Heaviside function
    for j in range(J+1):
        Psi = (  np.sqrt(2.0 * np.pi * s[j] / dt)
               * np.pi ** (-0.25)
               * np.exp(-((s[j]*omega - omega0) ** 2) / 2.0)
               * Hev )
        cwt[j, :] = ifft(series_ft * np.conjugate(Psi))

    cwt = cwt[:, :len(series)] # remove padding
    s_to_f = (omega0 + np.sqrt(2 + omega0**2)) / (4.0*np.pi)
    freq = s_to_f / s  # frequency corresponding to scale s

    return freq, cwt

def calc_coi(run):
    """
    arguments: run -- Run object
    return: coi -- (Nt,) cone of influence of the wavelet transform
    """
    omega0 = 6.0
    s_to_f = (omega0 + np.sqrt(2 + omega0**2)) / (4.0*np.pi)
    dt = run.delt * run.ifdiag
    coi = np.zeros_like(run.time)
    coi[0] = 0.5 / dt
    coi[1:run.Nt//2] = np.sqrt(2) * s_to_f / (run.time[1:run.Nt//2] - run.time[0])
    coi[run.Nt//2:-1] = np.sqrt(2) * s_to_f / (run.time[-1] - run.time[run.Nt//2:-1])
    coi[-1] = 0.5 / dt
    return coi

def inverse_wavelet_transform(run, cwt):
    """
    arguments: run -- Run object
               cwt -- (J+1, Nt) wavelet transform coefficients
    return: series -- (Nt,) reconstructed time series data
    """
    # constants for Morlet wavelet
    Nt = cwt.shape[1]
    n_cwt = int(2**(np.ceil(np.log2(Nt))))
    dt = run.delt * run.ifdiag
    dj = 0.125
    omega0 = 6.0
    s0 = 2 * dt

    J = cwt.shape[0] - 1
    s = s0 * 2**(dj * np.arange(0, J+1, 1))
    omega = 2 * np.pi * fftfreq(n_cwt, dt)
    
    cwt_padded = np.pad(cwt, ((0,0), (0,n_cwt-Nt)), 'constant')
    # reconstruction
    series_ft_rec = np.zeros(n_cwt, dtype=complex)
    Hev = np.array( omega > 0.0, dtype=float ) # Heaviside function
    for j in range(J+1):
        Psi = (  np.sqrt(2.0 * np.pi*s[j]/dt)
               * np.pi ** (-0.25)
               * np.exp(-((s[j]*omega - omega0) ** 2) / 2.0)
               * Hev )
        scale_factor = dj / s[j] * dt / (0.776) # 0.776: for Morlet wavelet with omega0=6
        series_ft_rec += fft(cwt_padded[j,:], n=n_cwt) * Psi * scale_factor

    series_rec = ifft(series_ft_rec).real # into time domain
    series_rec = series_rec[:Nt]

    return series_rec

def draw_power_spectrum(run, series, fig=None, ax=None,
                        flog=True, fmin=1e-3, fmax=22e-3, vmax=None,
                        cmap='jet', fontsize=20, unit='unit', label=''):
    """
    arguments: run    -- Run object (data must be read at every time step)
               series -- (Nt,) time series data
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    
    freq, cwt = wavelet_transform(run, series)
    power = np.abs(cwt)**2
    coi = calc_coi(run)

    if vmax is None:
        vmax = np.log10(power[(freq>=fmin) & (freq<=fmax), :]).max()
    norm = mpl.colors.Normalize(vmin=vmax-9, vmax=vmax)
    cmap = plt.get_cmap(cmap)

    dt = run.delt * run.ifdiag
    x = np.linspace(run.time[0] - dt/2, run.time[-1] + dt/2, run.Nt+1)
    y = np.zeros(len(freq)+1)
    y[0] = (3*freq[0] - freq[1]) / 2
    y[1:-1] = (freq[:-1] + freq[1:]) / 2
    y[-1] = (3*freq[-1] - freq[-2]) / 2

    ax.pcolormesh(x, y, np.log10(power), norm=norm, cmap=cmap)
    ax.fill_between(run.time, coi, fc='w', alpha=0.5)

    position = ax.get_position()
    cbar_ax = fig.add_axes([position.x1+0.02, position.y0, 0.02, position.height])
    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
    cbar.set_label(f'log$_{{10}}$(({unit})$^2)$', fontsize=11)

    ax.set_xlim(run.time[0], run.time[-1])
    ax.tick_params(labelbottom=True)
    ax.set_ylim(fmin, fmax)
    ax.set_yscale('log' if flog else None)
    ax.set_ylabel('Frequency [Hz]',fontsize=fontsize)

    text = ax.text(0.01, 0.05, label, color='white', fontsize=fontsize, fontweight='bold', transform=ax.transAxes)
    text.set_path_effects([patheffects.Stroke(linewidth=2, foreground='black'), patheffects.Normal()])    


if __name__ == "__main__":
    i2, i3 = 4, 20

    run = Run('../../run/case1b256')
    run.set_trange((0, 2161, 1))
    run.read_equatorial('bg')
    run.read_equatorial('field')

    B = run.B[i3, i2, :, :] + run.B0[i3, i2, :, None]
    V = run.V[i3, i2, ...]
    Ephi = -(V[0, :] * B[1, :] - V[1, :] * B[0, :])
    Ephi *= run.unitE / run.unitV / run.unitB
    series = bandpass_filter(run, Ephi, 1.6e-3, 7e-3)
    freq, cwt = wavelet_transform(run, series)
    series_rec = inverse_wavelet_transform(run, cwt)

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    axes[0].plot(run.time, series, label='original')
    axes[0].plot(run.time, series_rec, label='reconstructed', linestyle='dashed')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('E_phi [mV/m]')
    axes[0].legend()
    axes[0].set_xlim(0, 4000)
    draw_power_spectrum(run, series, fig=fig, ax=axes[1],
                        flog=True, fmin=1e-3, fmax=22e-3, vmax=None,
                        fontsize=15, unit='E_phi [mV/m]', label='E_phi at (i2,i3)=('+str(i2)+','+str(i3)+')')
    axes[1].set_xlim(0, 4000)

    Bpara = -B[0, :]
    series = bandpass_filter(run, Bpara, 1.6e-3, 7e-3)
    freq, cwt = wavelet_transform(run, series)
    series_rec = inverse_wavelet_transform(run, cwt)   
    axes[2].plot(run.time, series, label='original')
    axes[2].plot(run.time, series_rec, label='reconstructed', linestyle='dashed')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('B_para [nT]')
    axes[2].legend()
    axes[2].set_xlim(0, 4000)
    draw_power_spectrum(run, series, fig=fig, ax=axes[3],
                        flog=True, fmin=1e-3, fmax=22e-3, vmax=None,
                        fontsize=15, unit='B_para [nT]', label='B_para at (i2,i3)=('+str(i2)+','+str(i3)+')')
    axes[3].set_xlim(0, 4000)
    plt.tight_layout()
    plt.show()
