import os

import numpy as np
import matplotlib.pyplot as plt


savefile = 'input_boundary.pdf'

def read(filename):
    """
    Parameter
    ---------
    filename : run{irun}-00{Lvalue}.txt

    Returns
    -------
    data : Columns are
    L shell, MLT, temperature (eV), density (/cc), Bx (nT), By (nT), Bz(nT), Vx (km/s), Vy (km/s), and Vz (km/s)
       0   ,  1 ,       2         ,     3,       ,    4   ,   5    ,   6   ,    7     ,    8     ,        9
    """
    data = []
    with open(filename, 'r') as f:
        for l in f:
            data.append(list(map(float,l.split())))
        return np.array(data)

def pick_plasmasheet_value(data):
    """
    Parameter
    ---------
    data : output of read(filename)

    Returns
    -------
    temp, den : plasmasheet temperature (keV) and density (/cc)
    """
    ipeak = np.argmax(data[:,2])
    temp = data[ipeak,2] / 1e3  # keV
    den = data[ipeak,3]        # /cc
    return temp, den

# results from MHD simulations
prefix = '../input/plasma_sheet'
Lvalues = ['76', '66', '66']
filenames = [os.path.join(prefix, f'run{2*irun+1}-00{Lvalues[irun]}.txt') for irun in range(3)]

fig, axes = plt.subplots(2, 3, figsize=(12,8))
for irun in range(3):
    data = read(filenames[irun])
    mlt = data[:,1]
    temp = data[:,2] / 1e3 # keV
    den = data[:,3]

    temp_peak, den_peak = pick_plasmasheet_value(data)

    # plot mlt = -12 to 12
    mlt[mlt > 12] -= 24
    idx = np.argsort(mlt)
    mlt = mlt[idx]
    temp = temp[idx]
    den = den[idx]

    # fit functions
    N3 = 128 + 1
    x3 = np.linspace(0, 2*np.pi, N3) # GEMSIS-RC grid
    mlt_to_rad = 2*np.pi/24.0
    mlt_x3 = (x3-np.pi) / mlt_to_rad
    d1 = (18.0-12.0) * mlt_to_rad # MLT = 18h
    d2 = (6.0+12.0) * mlt_to_rad # MLT = 6h
    dm = 1.0 * mlt_to_rad # width
    fit_temp = 0.25 * temp_peak * (1+np.tanh((x3-d1)/dm)) * (1-np.tanh((x3-d2)/dm))
    fit_den  = 0.25 * den_peak * (1+np.tanh((x3-d1)/dm)) * (1-np.tanh((x3-d2)/dm))

    axes[0,irun].set_title(f'Case {irun+1}', fontsize=20)
    axes[0,irun].plot(mlt, temp, '-x', label='MHD simulation', color='black')
    axes[0,irun].plot(mlt_x3, fit_temp, '-x', color='red', label='Fit')
    axes[1,irun].set_xlabel('MLT [h]', fontsize=20)
    axes[0,irun].set_xlim([-12,12])
    axes[0,irun].set_ylim([0,10])
    axes[0,irun].tick_params(labelsize=16)

    axes[1,irun].plot(mlt, den, '-x', label='MHD simulation', color='black')
    axes[1,irun].plot(mlt_x3, fit_den, '-x', color='red', label='Fit')
    axes[1,irun].set_yscale('log')
    axes[1,irun].set_xlim([-12,12])
    axes[1,irun].set_ylim([1e-5, 1e2])
    axes[1,irun].tick_params(labelsize=16)

axes[1,0].set_ylabel('Density [/cc]', fontsize=20)
axes[0,0].set_ylabel('Temperature [keV]', fontsize=20)
axes[0,0].legend(loc='upper left')

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
for i, ax in enumerate(axes.flatten()):
    ax.text(-0.1, 1.13, f'$\mathbf{{{labels[i]}}}$', transform=ax.transAxes, fontsize=24, va='top')

plt.savefig(savefile, bbox_inches='tight')
