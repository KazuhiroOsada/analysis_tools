import os
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from chunk_reader import bg_reader, field_reader, dist_reader


def calc_vperp(run, B):
    """
    Parameters
    ----------
    run     : Run object
    B       : magnetic field strength [T]

    Returns
    -------
    vperp : perpendicular velocity [m/s], shape=(Nm,)
    """
    return np.sqrt(2 * run.mu * B * run.Qp / run.Mp)

def calc_bounce_and_drift_freqs(run, B, i2):
    """
    Hamlin (1961)'s approximation
    omega_b [/s] = pi*sqrt(W)/sqrt(2m)LRe * 1/(1.3-0.56sin(alpha))
    omega_d [/s] = -6W/qBL^2Re^2 (0.35+0.15sin(alpha)) (negative sign means westward drift)

    Parameters
    ----------
    run : Run object
    B   : magnetic field strength [T] at magnetic equator
    L   : L value

    Returns
    -------
    omega_b : bounce frequency [rad/s], shape=(Nm,Nv)
    omega_d : drift frequency [rad/s], shape=(Nm,Nv)
    """
    L = 1 / run.x2[i2]**2 / run.Re
    vperp = calc_vperp(run, B) # m/s
    Vpe, Vpa = np.meshgrid(vperp, run.vp, indexing='ij') # (Nm,Nv)
    W = 1/2 * run.Mp * (Vpe**2+Vpa**2) # J
    sinA = Vpe / np.sqrt(Vpe**2 + Vpa**2)
    wb = np.pi * np.sqrt(W) / (np.sqrt(2*run.Mp)*(L*run.Re)) * 1/(1.3-0.56*sinA)
    wd = -6 * W / (run.Qp*B*(L*run.Re)**2) * (0.35+0.15*sinA) # /s
    return wb, wd

def calc_dWdL(run, B, i2, omega, m):
    """
    dWdL = omega/m*q*Re^2BL (Southwood, 1969)

    Parameters
    ----------
    run     : Run object
    B       : magnetic field strength [T]
    i2      : global index for x2 direction
    omega   : wave angular frequency [rad/s]
    m       : azimuthal wave number

    Returns
    -------
    dWdL [J]
    """
    L = 1 / run.x2[i2]**2 / run.Re
    return omega / m * run.Qp * run.Re**2 * B * L

def calc_dfdW(run, dist, im, iv):
    """
    df/dW|_{mu, W} (Southwood, 1969)
    
    Parameters
    ----------
    run     : Run object
    dist    : distribution function [s^3/m^6], shape=(Nm, Nv)
    im, iv  : indices for mu and vpara

    Returns
    -------
    dfdW [s^3/m^6/J]
    """
    if iv == 0: # forward difference
        dW = 1/2 * run.Mp * (run.vp[iv+1]**2 - run.vp[iv]**2)
        return (dist[im, iv+1] - dist[im, iv]) / dW
    elif iv == run.Nv - 1: # backward difference
        dW = 1/2 * run.Mp * (run.vp[iv]**2 - run.vp[iv-1]**2)
        return (dist[im, iv] - dist[im, iv-1]) / dW
    else: # central difference
        dW = 1/2 * run.Mp * (run.vp[iv+1]**2 - run.vp[iv-1]**2)
        return (dist[im, iv+1] - dist[im, iv-1]) / dW
    
def calc_dfdL(run, dist_m, dist, dist_p, B_m, B, B_p, i2, im, iv):
    """
    df/dL|_{mu, W} (Southwood, 1969)
    
    Parameters
    ----------
    run                  : Run object
    dist_m, dist, dist_p : distribution function [s^3/m^6] at i2-1, i2, i2+1, shape=(Nm, Nv)
    B_m, B, B_p          : magnetic field strength [T] at i2-1, i2, i2+1
    i2                   : global index for x2 direction
    im, iv               : indices for mu and vpara

    Returns
    -------
    dfdL [s^3/m^6]

    Notes
    -----
    The calculation is not allowed at the boundaries of i2.
    """
    if i2 == 0 or i2 == run.N2 - 1:
        print(f'df/dL calculation at the x2 boundary (i2 = {i2}) is not allowed. Return nan.')
        return np.nan
    if iv == 0:
        print(f'df/dL calculation at the minimum vpara (iv = {iv}) is not allowed. Return nan.')
        return np.nan
    L = 1 / run.x2**2 / run.Re # (N2,)
    # find vpara at i2-1 and i2+1 for conservation of mu and W
    Wpara_p = run.mu[im] * (B - B_p) * run.Qp + 1/2 * run.Mp * run.vp[iv]**2
    Wpara_m = run.mu[im] * (B - B_m) * run.Qp + 1/2 * run.Mp * run.vp[iv]**2
    vpara_m = np.sqrt(2 / run.Mp * Wpara_m) * np.sign(run.vp[iv])
    # interpolate log(dist) at vpara_p and vpara_m
    eps = 1e-300 # to avoid log(0)
    if Wpara_p < 0: # not enough parallel energy to reach the point i2+1 (stronger B)
        iv_m = np.searchsorted(run.vp, vpara_m) # vp[iv_m-1] < vpara_m <= vp[iv_m]
        if iv_m < run.Nv:
            log_dist_m = ( np.log(dist_m[im, iv_m-1] + eps) * (run.vp[iv_m] - vpara_m)
                         + np.log(dist_m[im, iv_m] + eps) * (vpara_m - run.vp[iv_m-1]) ) / (run.vp[iv_m] - run.vp[iv_m-1])
            dist_m_interp = np.exp(log_dist_m)
        else:
            # in some cases, vpara_m is larger than the maximum vp, then use the value at the maximum vp
            dist_m_interp = dist_m[im, -1]
        return (dist[im, iv] - dist_m_interp) / (L[i2] - L[i2-1])
    else:
        vpara_p = np.sqrt(2 / run.Mp * Wpara_p) * np.sign(run.vp[iv])
        iv_p = np.searchsorted(run.vp, vpara_p) # vp[iv_p-1] < vpara_p <= vp[iv_p]
        iv_m = np.searchsorted(run.vp, vpara_m) # vp[iv_m-1] < vpara_m <= vp[iv_m]
        if iv_m < run.Nv: 
            log_dist_p = ( np.log(dist_p[im, iv_p-1] + eps) * (run.vp[iv_p] - vpara_p)
                         + np.log(dist_p[im, iv_p] + eps) * (vpara_p - run.vp[iv_p-1]) ) / (run.vp[iv_p] - run.vp[iv_p-1])
            log_dist_m = ( np.log(dist_m[im, iv_m-1] + eps) * (run.vp[iv_m] - vpara_m)
                         + np.log(dist_m[im, iv_m] + eps) * (vpara_m - run.vp[iv_m-1]) ) / (run.vp[iv_m] - run.vp[iv_m-1])
            dist_p_interp = np.exp(log_dist_p)
            dist_m_interp = np.exp(log_dist_m)
        else:
            # in some cases, vpara_m is larger than the maximum vp, then use the value at the maximum vp
            log_dist_p = ( np.log(dist_p[im, iv_p-1] + eps) * (run.vp[iv_p] - vpara_p)
                         + np.log(dist_p[im, iv_p] + eps) * (vpara_p - run.vp[iv_p-1]) ) / (run.vp[iv_p] - run.vp[iv_p-1])
            dist_p_interp = np.exp(log_dist_p)
            dist_m_interp = dist_m[im, -1]          
        return (dist_p_interp - dist_m_interp) / (L[i2+1] - L[i2-1])

def calc_gamma_velocity_space(run, dist_m, dist, dist_p, B_m, B, B_p,
                              i2, omega, m, n=0, threshold=0.5):
    """
    Calculation of growth rate on velocity space
    Implementation of the growth rate calculation following Southwood (1969)
    gamma1 = pi*mu0/3v^5/(2mwd+wb) * 1/(B^2R^2) * (dWdL)^2 * df/dW|_{mu,L}
    gamma2 = pi*mu0/3v^5/(2mwd+wb) * 1/(B^2R^2) *  dWdL    * df/dL|_{mu,W}
    gamma = gamma1 + gamma2
    
    Parameters
    ----------
    run                  : Run object
    dist_m, dist, dist_p : distribution function [s^3/m^6] at i2-1, i2, i2+1, shape=(Nm, Nv)
    B_m, B, B_p          : magnetic field strength [T] at i2-1, i2, i2+1
    i2                   : global index for x2 direction
    omega                : wave angular frequency [rad/s]
    m                    : azimuthal wave number
    n                    : harmonic number (default: 0 (fundamental))
    threshold            : when {(omega - m*wd - n*wb) / omega < threshold} is satisfied resonance is considered (default: 0.5)

    Returns
    -------
    gamma1              : growth rate component from df/dW [1/s], shape=(Nm,Nv)
    gamma2              : growth rate component from df/dL [1/s], shape=(Nm,Nv)
    gamma               : total growth rate [1/s], shape=(Nm,Nv)
    resonance_condition : (omega - m*wd - n*wb) [rad/s], shape=(Nm,Nv)

    Note
    ----
    The calculation is not allowed at the boundaries of i2 and at the minimum vpara
    Setting of threshold do not need to so strict, because when the resonance condition is not well satisfied, calculated gamma will be small
    """
    vperp = calc_vperp(run, B)
    wb, wd = calc_bounce_and_drift_freqs(run, B, i2) # (Nm, Nv), (Nm, Nv)
    resonance_condition = omega - m * wd - n * wb # (Nm, Nv)
    dWdL = calc_dWdL(run, B, i2, omega, m)
    gamma1, gamma2, gamma = np.full((run.Nm, run.Nv), np.nan), np.full((run.Nm, run.Nv), np.nan), np.full((run.Nm, run.Nv), np.nan)
    for im in range(run.Nm):
        for iv in range(1,run.Nv):
            if np.abs(resonance_condition[im, iv]) < threshold * omega:
                dfdW = calc_dfdW(run, dist, im, iv)
                dfdL = calc_dfdL(run, dist_m, dist, dist_p, B_m, B, B_p, i2, im, iv)
                prefactor = (( np.pi * run.mu0 / 3) * np.sqrt(vperp[im]**2 + run.vp[iv]**2)**5 
                            / (2 * m * wd[im, iv] + wb[im, iv])
                            / (B**2 * run.Re**2))                  
                gamma1[im, iv] = prefactor * dWdL**2 * dfdW
                gamma2[im, iv] = prefactor * dWdL    * dfdL
                gamma[im, iv] = gamma1[im, iv] + gamma2[im, iv]
    return gamma1, gamma2, gamma, resonance_condition

def draw_gamma_velocity_space(run, dist_m, dist, dist_p, B_m, B, B_p,
                              i2, omega, m, n=0, threshold=0.5, title='', filename=None):
    """
    Make a summary plot of growth rate on velocity space
    axes[0] : distribution function
    axes[1] : growth rate
    axes[2] : growth rate component from df/dW (gamma1)
    axes[3] : resonance condition

    Parameters
    ----------
    Same as calc_gamma_velocity_space

    Returns
    -------
    Same as calc_gamma_velocity_space
    """
    from draw_psd import draw_on_velocity_space

    gamma1, gamma2, gamma, rsc = calc_gamma_velocity_space(run, dist_m, dist, dist_p, B_m, B, B_p,
                                                      i2, omega, m, n, threshold)
    
    fig, axes = plt.subplots(1, 4, figsize=(18,6))
    fig.suptitle(title + f', $\gamma$ = {np.nanmax(gamma):.5f} /s, threshold = {threshold:.2f}', fontsize=16)
    draw_on_velocity_space(run, dist, xaxis='vperp', B=B*run.unitB, fig=fig, ax=axes[0], title='PSD [s$^3$/m$^6$]',)
    draw_on_velocity_space(run, gamma, xaxis='vperp', B=B*run.unitB, fig=fig, ax=axes[1],
                           cmap='coolwarm', vmin=-0.003, vmax=0.003)
    draw_on_velocity_space(run, (gamma == np.nanmax(gamma)).astype(float), xaxis='vperp', B=B*run.unitB, title='$\gamma = \gamma_1 + \gamma_2$',
                           fig=fig, ax=axes[1], cmap='binary', vmin=0, vmax=1, alpha=0.3, colorbar=False)
    draw_on_velocity_space(run, gamma1, xaxis='vperp', B=B*run.unitB,
                           fig=fig, ax=axes[2], title='$\gamma_1 \propto df/dW$',
                           cmap='coolwarm', vmin=-0.003, vmax=0.003)
    draw_on_velocity_space(run, rsc/omega, xaxis='vperp', B=B*run.unitB,
                           fig=fig, ax=axes[3], title='Resonance condition', cmap='coolwarm', vmin=-1.0, vmax=1.0)
    draw_on_velocity_space(run, (np.abs(rsc) < threshold * omega).astype(float),
                           xaxis='vperp', B=B*run.unitB, fig=fig, ax=axes[3], cmap='binary', vmin=0, vmax=1, alpha=0.1, colorbar=False, title='$|\omega - m\omega_d|/\omega$')
    # add drift frequency contours
    for ax in axes:
        _, wd = calc_bounce_and_drift_freqs(run, B, i2)
        vperp = calc_vperp(run, B) * 1e-3 # to km/s
        x, y = np.meshgrid(vperp, run.vp*1e-3, indexing='ij')
        ctr = ax.contour(x, y, wd*1e3/(2*np.pi), colors='gray', linestyles='dashed')
        ax.clabel(ctr, fmt='%.2f', colors='gray', fontsize=8)

    if filename is not None:
        fig.savefig(filename)
    else:
        plt.show()
    plt.close(fig)

    return gamma1, gamma2, gamma, rsc

def calc_gammas(run, cwt_analysis_file, i3, i2, n=0, threshold=0.5, plot=True):
    """
    Calculate time series of growth rate

    Parameters
    ----------
    run                 : Run object with coord data and trange_v set
    cwt_analysis_file   : path to .npz file created by cwt_analysis.py
    i3, i2              : indices in x3 and x2 directions
    n                   : harmonic number (default: 0 (fundamental))
    threshold           : when {(omega - m*wd - n*wb) / omega < threshold} is satisfied resonance is considered (default: 0.5)
    plot                : if True, make summary plots to visualize the calculation (default: False)

    Returns
    -------
    time     : time array [s], shape=(Nt_v,)
    gamma1s  : growth rate component from df/dW [1/s], shape=(Nt_v,)
    gamma2s  : growth rate component from df/dL [1/s], shape=(Nt_v,)
    gammas   : total growth rate [1/s], shape=(Nt_v,)
    vperps   : resonant perpendicular velocity [m/s], shape=(Nt_v,)
    vparas   : resonant parallel velocity [m/s], shape=(Nt_v,)
    res_dist : resonant PSD [s^3/m^6], shape=(Nt_v,)
    """
    i1 = run.N1//2 # equatorial plane
    d1, l1 = i1 // run.N1_local, i1 % run.N1_local
    d2, l2 = i2 // run.N2_local, i2 % run.N2_local
    d3, l3 = i3 // run.N3_local, i3 % run.N3_local

    file_path_dist = os.path.join(run.prefix, f'dist1-{d1:02d}-{d2:02d}-{d3:02d}.dat')
    d = dist_reader(file_path_dist, run.N1_local, run.N2_local, run.N3_local, run.Nm, run.Nv, run.trange_v)
    print(f'l2: {l2}')
    dist_p = d[l3, l2+1, l1, :, :, :] # (Nm, Nv, Nt)
    dist   = d[l3, l2,   l1, :, :, :] # (Nm, Nv, Nt)
    dist_m = d[l3, l2-1, l1, :, :, :] # (Nm, Nv, Nt)

    file_path_bg = os.path.join(run.prefix, f'bg-{d1:02d}-{d2:02d}-{d3:02d}.dat')
    B0, _ = bg_reader(file_path_bg, run.N1_local, run.N2_local, run.N3_local)
    file_path_field = os.path.join(run.prefix, f'field-{d1:02d}-{d2:02d}-{d3:02d}.dat')
    _, b = field_reader(file_path_field, run.N1_local, run.N2_local, run.N3_local, run.trange)
    B_m = np.linalg.norm(B0[l3, l2-1, l1, :, None] + b[l3, l2-1, l1, :, :], axis=0) # (Nt,)
    B   = np.linalg.norm(B0[l3, l2,   l1, :, None] + b[l3, l2,   l1, :, :], axis=0) # (Nt,)
    B_p = np.linalg.norm(B0[l3, l2+1, l1, :, None] + b[l3, l2+1, l1, :, :], axis=0) # (Nt,)

    data = np.load(cwt_analysis_file)
    freq = data['max_power_freq']
    mnums = data['mnumbers']

    gamma1s  = np.full(run.Nt_v, np.nan)
    gamma2s  = np.full(run.Nt_v, np.nan)
    gammas   = np.full(run.Nt_v, np.nan)
    vperps   = np.full(run.Nt_v, np.nan)
    vparas   = np.full(run.Nt_v, np.nan)
    res_dist = np.full(run.Nt_v, np.nan)
    for it in range(run.Nt_v):
        omega = 2*np.pi * freq[i3,i2,int(run.time_v[it]/(run.ifdiag*run.delt))]
        m     = mnums[i3,i2,int(run.time_v[it]/(run.ifdiag*run.delt))]
        if np.isnan(omega) or np.isnan(m):
            continue
        if not plot:
            gamma1, gamma2, gamma, _ = calc_gamma_velocity_space(run, dist_m[..., it], dist[..., it], dist_p[..., it],
                                                                 B_m[it], B[it], B_p[it], i2, omega, m, n, threshold)
        else:
            if not os.path.exists('gamma_plots'):
                os.makedirs('gamma_plots')
            gamma1, gamma2, gamma, _ = draw_gamma_velocity_space(run, dist_m[..., it], dist[..., it], dist_p[..., it],
                                                                 B_m[it], B[it], B_p[it], i2, omega, m, n, threshold,
                                                                 title=f't={run.time_v[it]:.1f}s, f={omega/(2*np.pi)*1e3:.2f} mHz, m={m:.2f}',
                                                                 filename=f'gamma_plots/{run.prefix.name}_{it:02d}.png')
        try:
            im, iv = np.unravel_index(np.nanargmax(gamma), gamma.shape)
            gammas[it] = gamma[im, iv]
            gamma1s[it] = gamma1[im, iv]
            gamma2s[it] = gamma2[im, iv]
            vperps[it] = calc_vperp(run, B[it])[im] * run.unitV
            vparas[it] = run.vp[iv] * run.unitV
            res_dist[it] = dist[im, iv, it]
        except:
            pass
    return run.time_v, gamma1s, gamma2s, gammas, vperps, vparas, res_dist

def main():
    i3, i2 = 5, 4

    rundirs = ['case1b128', 'case2b128', 'case3b128']
    fig, axes = plt.subplots(5, 1, figsize=(10,15), sharex=True)

    for i in range(3):
        cwt_file = f'cwt/analysis_{rundirs[i]}_Ephi.npz'
        data = np.load(cwt_file)
        powers = data['max_power']
  
        run = Run(f'../../run/{rundirs[i]}')
        run.read('coord')
        trange_v = (0, 91, 1)
        run.set_trange(trange_v, 'v')
        run.read_equatorial('moment')
        time, gamma1s, gamma2s, gammas, vperps, vparas, res_dist = calc_gammas(run, cwt_file, i3, i2, n=0, threshold=0.5, plot=True)
        colors = ['tab:blue', 'tab:orange', 'tab:green']
        axes[0].plot(np.arange(2161)*5.0, np.log10(powers[i3,i2,:]), label=f'Case {i+1}', color=colors[i])
        axes[1].plot(time, gammas, color=colors[i])
        axes[1].plot(time, gamma1s, linestyle='dashed', color=colors[i])
        axes[1].plot(time, gamma2s, linestyle='dotted', color=colors[i])
        axes[2].plot(time, vperps, linestyle='solid', color=colors[i])
        axes[2].plot(time, vparas, linestyle='dashed', color=colors[i])
        axes[3].plot(time, res_dist, label=f'Case {i+1}', color=colors[i])
        axes[4].plot(time, run.Ppe[i3,i2,:], label=f'Case {i+1}', color=colors[i])
    axes[0].set_ylabel('log$_{10}$(Power [(mV/m)$^2$])')
    axes[0].legend()
    axes[1].set_ylabel('Growth rate [1/s]')
    labels_gamma = {'solid': '$\gamma$', 'dashed': '$\gamma_1$', 'dotted': '$\gamma_2$'}
    labels_velocity = {'solid': 'v$_\perp$', 'dashed': 'v$_\parallel$'}
    for ls in ['solid', 'dashed', 'dotted']:
        axes[1].plot([], [], linestyle=ls, color='black', label=labels_gamma[ls])
    axes[1].legend()
    for ls in ['solid', 'dashed']:
        axes[2].plot([], [], linestyle=ls, color='black', label=labels_velocity[ls])
    axes[2].legend()
    axes[3].set_ylabel('Resonant PSD [s$^3$/m$^6$]')
    axes[4].set_ylabel('Pressure [Pa]')
    axes[4].set_xlabel('Time [s]')
    axes[4].set_xlim(600, 4000)
    for ax in axes:
        ax.grid()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    i3, i2 = 5, 4
    run = Run('../../run/case1b128')
    run.set_trange((0,91,1), 'v')
    d1, d2, d3, l1, l2, l3 = run.resolve_global_idx(32, i2, i3)
    dist_file = os.path.join(run.prefix, f'dist1-{d1:02d}-{d2:02d}-{d3:02d}.dat')
    dist = dist_reader(dist_file, run.N1_local, run.N2_local, run.N3_local, run.Nm, run.Nv, run.trange_v)
    dist = dist[l3, l2, l1, :, run.Nv//2, :] # (Nm, Nt)
    for it in range(run.Nt_v):
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(run.mu, dist[:, it])
        ax.set_ylim(1e-20, 1e-15)
        ax.set_xlabel('mu [eV/T]')
        ax.set_ylabel('PSD [s$^3$/m$^6$]')
        ax.set_title(f't={run.time_v[it]:.1f}s')
        plt.savefig(f'psdcase1/{it:02d}.png')
        plt.close(fig)

    