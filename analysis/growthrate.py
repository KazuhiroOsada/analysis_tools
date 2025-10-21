import os
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from chunk_reader import bg_reader, field_reader, dist_reader


class const:
    Qp = 1.602e-19 # proton charge [C]
    Mp = 1.673e-27 # proton mass [kg]
    Re = 6.371e6   # Earth radius [m]

def convert_mu_to_vperp(mu, B):
    """
    arguments: mu -- magnetic moment [eV/T]
               B  -- magnetic field [T]
    return   : vperp -- perpendicular velocity [m/s]
    """
    return np.sqrt(2 * mu * B * const.Qp / const.Mp)

def calc_bounce_drift_freq(vperp, vpara, B, L):
    """
    return : omega_d [/s] = -6W/qBL^2Re^2 (0.35+0.15sin(alpha))
    negative sign means westward drift
    see Hamlin 1961 for derivation
    """
    W = 1/2 * const.Mp * (vperp**2+vpara**2) # J
    sinA = vperp / np.sqrt(vperp**2+vpara**2)
    wb = np.pi * np.sqrt(W) / (np.sqrt(2*const.Mp)*(L*const.Re)) * 1/(1.3-0.56*sinA)
    wd = -6 * W / (const.Qp*B*(L*const.Re)**2) * (0.35+0.15*sinA) # /s
    return wb, wd

def calc_resonance_condition(run, B, L, omega, m, n):
    """
    B [T]
    calculate drift-bounce resonance condition on velocity space
    return : omega - m*omega_d - n*omega_b (~0 if resonance condition is satisfied)
    """
    vperp = convert_mu_to_vperp(run.mu, B) # m/s
    Vpe, Vpa = np.meshgrid(vperp, run.vpara)
    wb, wd = calc_bounce_drift_freq(Vpe, Vpa, B, L)
    return omega - m*wd - n*wb # /s

def get_dWdL(B, L, omega, m):
    """
    return dW/dL [J] = omega/mqRe^2BL (Southwood, 1969)
    """
    return omega / m * const.Qp * const.Re**2 * B * L

def get_dfdW(run, dist, i3, i2, im, iv, it):
    """
    return : df/dW|_{mu,L}
    """
    if iv == 0:
        # forward difference
        dW = 1/2*const.Mp*(run.vpara[iv+1]**2 - run.vpara[iv]**2) # J
        dfdW = (dist[i3,i2,i1,im,iv+1,it] - dist[i3,i2,i1,im,iv,it]) / dW # s^3/m^6/J
    elif iv == run.Nv-1:
        # backward difference
        dW = 1/2*const.Mp*(run.vpara[iv]**2 - run.vpara[iv-1]**2) # J
        dfdW = (dist[i3,i2,i1,im,iv,it] - dist[i3,i2,i1,im,iv-1,it]) / dW # s^3/m^6/J
    else:
        # central difference
        dW = 1/2*const.Mp*(run.vpara[iv+1]**2 - run.vpara[iv-1]**2) # J
        dfdW = (dist[i3,i2,i1,im,iv+1,it] - dist[i3,i2,i1,im,iv-1,it]) / dW # s^3/m^6/J
    return dfdW
    
def get_dfdL(run, dist, B, i3, i2, im, iv, it):
    """
    dist -- (N3_local, N2_local, N1_local, Nm, Nv, Nt) distribution function [s^3/m^6]
    B    -- (N3_local, N2_local, N1_local) magnetic field [T]
    i3, i2, im, iv, it : indices for dist
    return : df/dL|_{mu,W}
    """
    # assert i2 != 0 and i2 != self.N2-1 and iv != 0 and iv != self.Nv-1, "not calculate dfdL at the boundary"
    # find vpara at i2+1, i2-1 so that mu and W are conserved
    # _plus and _minus are values at i2+1 and i2-1
    vpara_plus = np.sqrt(2/const.Mp*(
                            (run.mu[im]*(B[i3, i2, i1] - B[i3, i2+1, i1])*const.Qp)
                            + 1/2*const.Mp*run.vpara[iv]**2)
                            )*np.sign(run.vpara[iv]) # km/s
    vpara_minus = np.sqrt(2/const.Mp*(
                            (run.mu[im]*(B[i3, i2, i1] - B[i3, i2-1, i1])*const.Qp)
                            + 1/2*const.Mp*run.vpara[iv]**2)
                            )*np.sign(run.vpara[iv]) # km/s
    # find the indices of the nearest neighbors
    # vpara[iv_plus-1] < vpara_plus <= vpara[iv_plus], vpara[iv_minus-1] < vpara_minus <= vpara[iv_minus]
    # get weighted log(psd) by linear interpolation
    if np.isnan(vpara_plus):
        # in case dW_perp exceeds W_para, vpara_plus^2 become less than 0
        # then dfdL is calculated by difference between i2 and i2-1
        # nearest neighbors
        iv_minus = np.searchsorted(run.vpara,vpara_minus)
        # weighted psd
        log_psd_minus = ( np.log(dist[i3, i2-1, im, iv_minus-1, it])*(run.vpara[iv_minus]-vpara_minus) 
                        + np.log(dist[i3, i2-1, im, iv_minus, it])*(vpara_minus-run.vpara[iv_minus-1]) 
                        ) / (run.vpara[iv_minus]-run.vpara[iv_minus-1])
        psd_minus = np.exp(log_psd_minus)
        dfdL = (dist[i3, i2, im, iv, it] - psd_minus) / (run.L[i2] - run.L[i2-1]) # s^3/m^6
    else:
        # nearest neighbors
        iv_plus = np.searchsorted(run.vpara,vpara_plus)
        iv_minus = np.searchsorted(run.vpara,vpara_minus)
        # weighted psd
        log_psd_plus  = ( np.log(dist[i3, i2+1, im, iv_plus-1, it])*(run.vpara[iv_plus]-vpara_plus) 
                        + np.log(dist[i3, i2+1, im, iv_plus, it])*(vpara_plus-run.vpara[iv_plus-1]) 
                        ) / (run.vpara[iv_plus]-run.vpara[iv_plus-1])
        log_psd_minus = ( np.log(dist[i3, i2-1, im, iv_minus-1, it])*(run.vpara[iv_minus]-vpara_minus) 
                        + np.log(dist[i3, i2-1, im, iv_minus, it])*(vpara_minus-run.vpara[iv_minus-1]) 
                        ) / (run.vpara[iv_minus]-run.vpara[iv_minus-1])
        psd_plus = np.exp(log_psd_plus)
        psd_minus = np.exp(log_psd_minus)
        dfdL = (psd_plus - psd_minus) / (run.L[i2+1] - run.L[i2-1]) # s^3/m^6
    return dfdL

def calc_gamma(run, dist, B, i3, i2 ,im, iv, it):
    """
    dist -- (N3_local, N2_local, Nm, Nv, Nt) distribution function [s^3/m^6]
    B    -- (N3_local, N2_local, Nt) magnetic field [T]
    """
    vperp = convert_mu_to_vperp(run.mu, B[i3, i2, it]) # m/s




if __name__ == '__main__':
    i1, i2, i3 = 32, 4, 200

    run = Run('../../run/case1b256')
    run.read('bg')
    trange_v = (0, 101, 5)
    run.set_trange(trange_v, 'v')

    d1, l1 = i1 // run.N1_local, i1 % run.N1_local
    d2, l2 = i2 // run.N2_local, i2 % run.N2_local
    d3, l3 = i3 // run.N3_local, i3 % run.N3_local

    file_path_dist = os.path.join(run.prefix, f'dist1-{d1:02d}-{d2:02d}-{d3:02d}.dat')
    dist = dist_reader(file_path_dist, run.N1_local, run.N2_local, run.N3_local, run.Nm, run.Nv, trange_v)
    dist = dist[l3, l2, l1, :, :, :] # (Nm, Nv, Nt)
    file_path_field = os.path.join(run.prefix, f'field-{d1:02d}-{d2:02d}-{d3:02d}.dat')
    V, B = field_reader(file_path_field, run.N1_local, run.N2_local, run.N3_local, run.trange)
    B = B[l3, l2, l1, :, :] # (3, Nt)
    B *= run.unitB # to [nT]
    Babs = np.zeros(run.Nt_v)
    for it in range(run.Nt_v):
        Babs[it] = np.linalg.norm(B[:, it] + run.B0[i3, i2, i1, :]) # [nT]