import os
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from chunk_reader import bg_reader, field_reader, dist_reader
from draw_cwt_analysis import read_cwt_analysis_data


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
    B   : magnetic field strength [T]
    L   : L value

    Returns
    -------
    omega_b : bounce frequency [/s], shape=(Nm,Nv)
    omega_d : drift frequency [/s], shape=(Nm,Nv)
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
    """
    L = 1 / run.x2**2 / run.Re # (N2,)
    # find vpara at i2-1 and i2+1 for conservation of mu and W
    Wpara_p = run.mu[im] * (B - B_p) * run.Qp + 1/2 * run.Mp * run.vp[iv]**2
    Wpara_m = run.mu[im] * (B - B_m) * run.Qp + 1/2 * run.Mp * run.vp[iv]**2
    vpara_m = np.sqrt(2 / run.Mp * Wpara_m) * np.sign(run.vp[iv])
    # interpolate log(dist) at vpara_p and vpara_m
    if Wpara_p < 0: # not enough parallel energy to reach the point i2+1 (stronger B)
        iv_m = np.searchsorted(run.vp, vpara_m) # vp[iv_m-1] < vpara_m <= vp[iv_m]
        log_dist_m = ( np.log(dist_m[im, iv_m-1]) * (run.vp[iv_m] - vpara_m)
                     + np.log(dist_m[im, iv_m]) * (vpara_m - run.vp[iv_m-1]) ) / (run.vp[iv_m] - run.vp[iv_m-1])
        dist_m = np.exp(log_dist_m)
        return (dist[im, iv] - dist_m) / (L[i2] - L[i2-1])
    else:
        vpara_p = np.sqrt(2 / run.Mp * Wpara_p) * np.sign(run.vp[iv])
        iv_p = np.searchsorted(run.vp, vpara_p) # vp[iv_p-1] < vpara_p <= vp[iv_p]
        iv_m = np.searchsorted(run.vp, vpara_m) # vp[iv_m-1] < vpara_m <= vp[iv_m]
        log_dist_p = ( np.log(dist_p[im, iv_p-1]) * (run.vp[iv_p] - vpara_p)
                     + np.log(dist_p[im, iv_p]) * (vpara_p - run.vp[iv_p-1]) ) / (run.vp[iv_p] - run.vp[iv_p-1])
        log_dist_m = ( np.log(dist_m[im, iv_m-1]) * (run.vp[iv_m] - vpara_m)
                     + np.log(dist_m[im, iv_m]) * (vpara_m - run.vp[iv_m-1]) ) / (run.vp[iv_m] - run.vp[iv_m-1])
        dist_p = np.exp(log_dist_p)
        dist_m = np.exp(log_dist_m)
        return (dist_p - dist_m) / (L[i2+1] - L[i2-1])

def calc_growth_rate(run, dist_m, dist_p, dist, B_m, B, B_p, i2, omega, m, n=0, threshold=1e-2):
    """
    gamma1 = pimu0/3 v^5/(2mwd+wb)1/(B^2R^2) (dWdL)^2 df/dW|_{mu,L}
    gamma2 = gamma2 = pi*mu0/3v^5/(2mwd+wb) * 1/(B^2R^2) * dWdL * df/dL|_{mu,W}
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
    threshold            : when (omega - m*wd - n*wb) / omega < threshold resonance is considered (default: 1e-2)

    Returns
    -------
    gamma1 : growth rate component from df/dW [1/s]
    gamma2 : growth rate component from df/dL [1/s]
    gamma  : total growth rate [1/s]
    indices: (im, iv) indices where the maximum growth rate occurs
    """
    vperp = calc_vperp(run, B)
    wb, wd = calc_bounce_and_drift_freqs(run, B, i2) # (Nm, Nv), (Nm, Nv)
    resonance_condition = omega - m * wd - n * wb # (Nm, Nv)
    from draw_psd import draw_on_velocity_space
    draw_on_velocity_space(run, resonance_condition)
    dWdL = calc_dWdL(run, B, i2, omega, m)
    gamma1, gamma2, gamma = 0.0, 0.0, 0.0
    indices = (-999, -999)
    for im in range(1,run.Nm-1):
        for iv in range(1,run.Nv-1):
            if resonance_condition[im, iv] < threshold * omega:
                dfdW = calc_dfdW(run, dist, im, iv)
                dfdL = calc_dfdL(run, dist_m, dist, dist_p, B_m, B, B_p, i2, im, iv)
                prefactor = (( np.pi * run.mu0 / 3) * np.sqrt(vperp[im]**2 + run.vp[iv]**2)**5 
                            / (2 * m * wd[im, iv] + wb[im, iv])
                            / (B**2 * run.Re**2))
                g1 = prefactor * dWdL**2 * dfdW
                g2 = prefactor * dWdL * dfdL
                g = g1 + g2
                if g > gamma:
                    gamma1 = g1
                    gamma2 = g2
                    gamma = g
                    indices = (im, iv)
    return gamma1, gamma2, gamma, indices
                

if __name__ == '__main__':
    from draw_psd import draw_on_velocity_space

    i1, i2, i3 = 32, 4, 40

    run = Run('../../run/case1b256')
    run.read('coord')
    trange_v = (20, 31, 1)
    run.set_trange(trange_v, 'v')

    d1, l1 = i1 // run.N1_local, i1 % run.N1_local
    d2, l2 = i2 // run.N2_local, i2 % run.N2_local
    d3, l3 = i3 // run.N3_local, i3 % run.N3_local

    file_path_dist = os.path.join(run.prefix, f'dist1-{d1:02d}-{d2:02d}-{d3:02d}.dat')
    dist = dist_reader(file_path_dist, run.N1_local, run.N2_local, run.N3_local, run.Nm, run.Nv, trange_v)
    dist_i2p = dist[l3, l2+1, l1, :, :, :] # (Nm, Nv, Nt)
    dist_i2  = dist[l3, l2, l1, :, :, :]   # (Nm, Nv, Nt)
    dist_i2m = dist[l3, l2-1, l1, :, :, :] # (Nm, Nv, Nt)

    file_path_bg = os.path.join(run.prefix, f'bg-{d1:02d}-{d2:02d}-{d3:02d}.dat')
    B0, _ = bg_reader(file_path_bg, run.N1_local, run.N2_local, run.N3_local)
    file_path_field = os.path.join(run.prefix, f'field-{d1:02d}-{d2:02d}-{d3:02d}.dat')
    _, B = field_reader(file_path_field, run.N1_local, run.N2_local, run.N3_local, run.trange)
    B_i2p = np.linalg.norm(B0[l3, l2+1, l1, :, None] + B[l3, l2+1, l1, :, :], axis=0) # (Nt,)
    B_i2  = np.linalg.norm(B0[l3, l2, l1, :, None]   + B[l3, l2, l1, :, :], axis=0)   # (Nt,)
    B_i2m = np.linalg.norm(B0[l3, l2-1, l1, :, None] + B[l3, l2-1, l1, :, :], axis=0) # (Nt,)

    max_power_freq, max_power, mnumbers = read_cwt_analysis_data('../../cwt/cwt_analysis_case1_Pc5_Ephi.npz')

    for it in range(run.Nt_v):
        print(f'Time = {run.time_v[it]:.1f} s')
        power = max_power[i3, i2, int(run.time_v[it] / 5.0)]
        omega = 2 * np.pi * max_power_freq[i3, i2, int(run.time_v[it] / 5.0)]
        m = np.abs(mnumbers[i3, i2, int(run.time_v[it] / 5.0)])
        print(f'power = {np.log10(power):.3e}, omega = {omega:.3e} rad/s, m = {m}')
        g1, g2, g, (im, iv) = calc_growth_rate(
            run,
            dist_i2m[:, :, it],
            dist_i2p[:, :, it],
            dist_i2[:, :, it],
            B_i2m[it],
            B_i2[it],
            B_i2p[it],
            i2,
            omega,
            m,
            n=0,
            threshold=2000
        )
        print(f'Growth rates: gamma1 = {g1:.3e} 1/s, gamma2 = {g2:.3e} 1/s, total gamma = {g:.3e} 1/s at im={im}, iv={iv}') 