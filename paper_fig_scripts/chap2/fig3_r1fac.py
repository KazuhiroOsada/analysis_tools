import os
import sys
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

from base import Run
from draw import draw_equatorial
from analysis.calc_potential import calc_potential


savefile = 'input_summary.pdf'
Ntheta, Nphi = 257, 257 # GEMSIS-POT grid size

def read_input_jp(filepath):
    theta = np.zeros(Ntheta)
    phi   = np.zeros(Nphi)
    jp    = np.zeros((Ntheta, Nphi))
    with open(filepath, 'r') as f:
        for i in range(Ntheta*Nphi):
            t_idx = i // Ntheta
            p_idx = i % Ntheta
            theta[t_idx], phi[p_idx], jp[t_idx, p_idx] = map(float, f.readline().split())
    return theta, phi, jp

def read_input_poten(filepath):
    potential = np.zeros((Ntheta, Nphi))
    with open(filepath, 'r') as f:
        for i in range(Ntheta*Nphi):
            t_idx = i // Ntheta
            p_idx = i % Ntheta
            potential[t_idx, p_idx] = float(f.readline().strip())
    return potential

def find_minmax_jp(theta, phi, jp):
    it_min, ip_min = np.unravel_index(np.argmin(jp), jp.shape)
    it_max, ip_max = np.unravel_index(np.argmax(jp), jp.shape)
    jp_min = jp[it_min, ip_min]
    jp_max = jp[it_max, ip_max]
    mlat_min = 90 - np.rad2deg(theta[it_min])
    mlat_max = 90 - np.rad2deg(theta[it_max])
    mlt_min = phi[ip_min] * 24 / (2*np.pi)
    mlt_max = phi[ip_max] * 24 / (2*np.pi)
    ic(jp_min, mlat_min, mlt_min)
    ic(jp_max, mlat_max, mlt_max)

def create_pcolormesh_grid(theta, phi):
    dt = theta[1] - theta[0]
    dp = phi[1] - phi[0]
    theta_edge = np.linspace(theta[0] - dt / 2, theta[-1] + dt / 2, len(theta) + 1)
    phi_edge = np.linspace(phi[0] - dp / 2, phi[-1] + dp / 2, len(phi) + 1)
    Th, Ph = np.meshgrid(theta_edge, phi_edge, indexing='ij')
    X = Th * np.cos(Ph)
    Y = Th * np.sin(Ph)
    return X, Y

def draw(theta, phi, Z, fig, ax, theta_max=np.pi/2, Lmin=None, Lmax=None ,vmin=None, vmax=None, colorbar=True, cmap='coolwarm'):
    # draw
    theta_mask = theta <= theta_max
    theta_north = theta[theta_mask]
    Z_north = Z[theta_mask, :]
    X, Y = create_pcolormesh_grid(theta_north, phi)
    pcm = ax.pcolormesh(X, Y, Z_north, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('equal')
    ax.set_xlim(-theta_max, theta_max)
    ax.set_ylim(-theta_max, theta_max)

    # axis
    ax.hlines(0, -theta_max, theta_max, color='gray', alpha=0.7, lw=0.7)
    ax.vlines(0, -theta_max, theta_max, color='gray', alpha=0.7, lw=0.7)
    angle = np.linspace(0, 2*np.pi, 100)
    t_axis = np.linspace(0, theta_max, 4)
    mlats_axis = 90 - np.rad2deg(t_axis) 
    mlats_label = [f'{mlat:.0f}°' for mlat in mlats_axis]
    fontsize = 14
    for t, mlat in zip(t_axis, mlats_label):
        ax.plot(t*np.cos(angle), t*np.sin(angle), color='gray', alpha=0.7, lw=0.7)
        ax.text(t*np.cos(np.pi/4), t*np.sin(np.pi/4), mlat, color='gray', fontsize=fontsize)
    ax.text(theta_max, 0, 'MLT  0', color='gray', ha='right', fontsize=fontsize)
    ax.text(0, theta_max, 'MLT  6', color='gray', va='top', fontsize=fontsize)
    ax.text(-theta_max, 0, 'MLT 12', color='gray', ha='left', fontsize=fontsize)
    ax.text(0, -theta_max, 'MLT 18', color='gray', va='bottom', fontsize=fontsize)
    ax.set_xticks([])
    ax.set_yticks([])

    # draw GEMSIS-RC range
    if Lmin is not None:
        Re = 6.378e6 # m
        h_iono = 100e3
        tmin = np.arcsin(np.sqrt((1+h_iono/Re)/Lmin))
        tmax = np.arcsin(np.sqrt((1+h_iono/Re)/Lmax))
        ax.plot(tmin*np.cos(t), tmin*np.sin(t), color='black', linestyle='dashed', alpha=0.7)
        ax.plot(tmax*np.cos(t), tmax*np.sin(t), color='black', linestyle='dashed', alpha=0.7)

    if colorbar:
        plt.colorbar(pcm, ax=ax)
    
    return pcm

def main():
    fig, axes = plt.subplots(3,3, figsize=(12,12))
    
    run_nums = [1, 3, 5]
    jp_files  = [f'../input/input-jp-run{run_num}.txt' for run_num in run_nums]
    pot_files = [f'../input/input-poten-run{run_num}.txt' for run_num in run_nums]
    prefix = '../../../run/'
    rundirs = [os.path.join(prefix, f'case{irun+1}b128') for irun in range(3)]

    for i in range(3):
        run = Run(rundirs[i])
        run.read_equatorial('coord')
        run.read_equatorial('bg')
        run.set_trange((0, 1, 1))
        run.read_equatorial('field')
        run.calc_electric_field()
        potential = calc_potential(run, run.E[..., 0])

        L = 1/run.x2**2/run.Re
        Lmin, Lmax = L[-1], L[0]
        theta, phi, jp = read_input_jp(jp_files[i])
        pot = read_input_poten(pot_files[i])
        print(jp_files[i])
        find_minmax_jp(theta, phi, jp)

        pcm_jp = draw(theta, phi, jp, fig, axes[0, i], vmin=-0.25, vmax=0.25, colorbar=False,
                      Lmin=Lmin, Lmax=Lmax, theta_max=np.pi/4)
        pcm_pot = draw(theta, phi, pot, fig, axes[1, i], vmin=-40, vmax=40, colorbar=False,
                       Lmin=Lmin, Lmax=Lmax, theta_max=np.pi/4)
        pcm_Rho0 = draw_equatorial(run, run.Rho0, fig, axes[2, i],
                                   vmin=10, vmax=1000, log=True, width=8.0, colorbar=False, ylabel=(i==0), return_pcm=True)
        axes[2, i].set_xticks(np.linspace(-8.0, 8.0, 9))
        axes[2, i].contour(run.Xi, run.Yi, potential, colors='white', linewidths=1.0, levels=np.arange(-100, 100, 10))
        axes[2, i].contour(run.Xi, run.Yi, potential, colors='white', linewidths=0.3, levels=np.arange(-100, 100, 2))
    
    lfs = 18
    axes[0,0].set_title('Case 1', fontsize=lfs+2)
    axes[0,1].set_title('Case 2', fontsize=lfs+2)
    axes[0,2].set_title('Case 3', fontsize=lfs+2)
    pos = axes[0,2].get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
    cbar = fig.colorbar(pcm_jp, cax=cbar_ax)
    pos = cbar_ax.get_position()
    cbar_ax.tick_params(labelsize=14)
    cbar_ax.text(0.3, 1.03, 'downward', transform=cbar_ax.transAxes, ha='center', fontsize=lfs-2)
    cbar_ax.text(0.3, -0.05, 'upward', transform=cbar_ax.transAxes, ha='center', fontsize=lfs-2)
    cbar.set_label('R1FAC [$\mathrm{\mu}$A/m$^2$]', fontsize=lfs)

    pos = axes[1,2].get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
    cbar = fig.colorbar(pcm_pot, cax=cbar_ax)
    cbar_ax.tick_params(labelsize=14)
    cbar.set_label('Electric potential [kV]', fontsize=lfs)

    pos = axes[2,2].get_position()

    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
    cbar = fig.colorbar(pcm_Rho0, cax=cbar_ax)
    cbar_ax.tick_params(labelsize=14)
    cbar.set_label('Density [/cc]', fontsize=lfs)

    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    for i, ax in enumerate(axes.flatten()):
        ax.text(-0.1, 1.13, f'$\mathbf{{{labels[i]}}}$', transform=ax.transAxes, fontsize=lfs+2, va='top')
    
    plt.savefig(savefile, bbox_inches='tight')


if __name__ == '__main__':
    main()
