import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from coordinate import ModifiedDipole


# constants
mu0 = 4.0 * np.pi * 1e-7
Me = -8.043e15 # T m^3 
Mp = 1.67e-27 # kg

def get_Rho0_Sheely(L, MLT=None, is_in_ps=True):
    """
    Sheely et al.(2001)
    """
    if is_in_ps: # in plasmasphere
        return 1390.0 * (3.0/L)**4.83 # /cc
    else: # in plasmatrough
        if MLT is None:
            raise ValueError("MLT must be provided for plasmatrough density calculation")
        return 124.0 * (3.0/L)**4.0 + 36 * (3.0/L)**3.5 * np.cos((MLT - (7.7 * (3.0/L)**2.0 + 12.0)) *np.pi/12.0) # /cc

def get_B0(r, theta, Me):
    return Me * r**(-3) * np.sqrt(4 - 3*np.sin(theta)**2)*1e9 # nT

def get_Va0(B0, Rho0):
    return np.abs(B0*1e-9) / np.sqrt(mu0 * Rho0*1e6*Mp) # m/s

def get_dx(coord, dphi):
    """
    Calculate the grid spacing in the modified dipole coordinate
    """
    dx1 = np.zeros((coord.N1-1, coord.N2))
    dx2 = np.zeros((coord.N1, coord.N2-1))
    dx3 = np.zeros((coord.N1, coord.N2))
    for i in range(coord.N1-1):
        for j in range(coord.N2):
            dx1[i, j] = np.sqrt((coord.x[i+1, j] - coord.x[i, j])**2 + (coord.z[i+1, j] - coord.z[i, j])**2) * coord.Re
    for i in range(coord.N1):
        for j in range(coord.N2-1):
            dx2[i, j] = np.sqrt((coord.x[i, j+1] - coord.x[i, j])**2 + (coord.z[i, j+1] - coord.z[i, j])**2) * coord.Re
    for i in range(coord.N1):
        dx3[i, :] = 2 * coord.x[i, :] * np.sin(dphi/2) * coord.Re
    return dx1, dx2, dx3

def get_Courant_number(dx1, dx2, dx3, dt, Va0):
    c1 = 0.5 * (Va0[:-1,:] + Va0[1:,:]) * dt / dx1
    c2 = 0.5 * (Va0[:,:-1] + Va0[:,1:]) * dt / dx2
    c3 = Va0 * dt / dx3
    return c1, c2, c3

def get_allowed_minimum_density(Va0_max, B0):
    """
    if Va0_max is set, one can find the allowed minimum density for each grid point
    """
    return (B0*1e-9 / Va0_max)**2 / (mu0 * Mp) * 1e-6 # /cc

def extend_half_integer_grid(coord, axis=0):
    """
    if axis=0, extend the grid in x2 direction
    if axis=1, extend the grid in x1 direction
    if axis=2, extend the grid in both directions
    """
    N1, N2 = coord.N1, coord.N2
    new_grid = [None, None]
    if axis == 0:
        for i, grid in enumerate([coord.x, coord.z]):
            new_grid[i] = np.zeros((N1, N2+1))
            new_grid[i][:, 0] = grid[:, 0] - (grid[:, 1] - grid[:, 0]) / 2
            new_grid[i][:, 1:-1] = (grid[:, :-1] + grid[:, 1:]) / 2
            new_grid[i][:, -1] = grid[:, -1] + (grid[:, -1] - grid[:, -2]) / 2
    elif axis == 1:
        for i, grid in enumerate([coord.x, coord.z]):
            new_grid[i] = np.zeros((N1+1, N2))
            new_grid[i][0, :] = grid[0, :] - (grid[1, :] - grid[0, :]) / 2
            new_grid[i][1:-1, :] = (grid[:-1, :] + grid[1:, :]) / 2
            new_grid[i][-1, :] = grid[-1, :] + (grid[-1, :] - grid[-2, :]) / 2
    elif axis == 2:
        for i, grid in enumerate([coord.x, coord.z]):
            tmp_grid = np.zeros((N1, N2+1))
            tmp_grid[:, 0] = grid[:, 0] - (grid[:, 1] - grid[:, 0]) / 2
            tmp_grid[:, 1:-1] = (grid[:, :-1] + grid[:, 1:]) / 2
            tmp_grid[:, -1] = grid[:, -1] + (grid[:, -1] - grid[:, -2]) / 2
            new_grid[i] = np.zeros((N1+1, N2+1))
            new_grid[i][0, :] = tmp_grid[0, :] - (tmp_grid[1, :] - tmp_grid[0, :]) / 2
            new_grid[i][1:-1, :] = (tmp_grid[:-1, :] + tmp_grid[1:, :]) / 2
            new_grid[i][-1, :] = tmp_grid[-1, :] + (tmp_grid[-1, :] - tmp_grid[-2, :]) / 2
    return new_grid

                
if __name__ == "__main__":
    Lmin = 3.6
    Lmax = 7.6
    Rmin = 3.0
    theta_min = np.pi/3
    N1 = 64
    N2 = 42
    N3 = 256
    dphi = 2 * np.pi / N3
    stretch = 100

    coord = ModifiedDipole(Lmin, Lmax, Rmin, theta_min, N1, N2, stretch)
    coord.L = coord.x2**(-2) / coord.Re
    Rho0 = get_Rho0_Sheely(coord.L, MLT=6.0, is_in_ps=False)
    print(Rho0)
    B0 = get_B0(coord.r*coord.Re, coord.theta, Me)
    Va0 = get_Va0(B0, Rho0)
    Va0_max = np.max(Va0)
    i1_Va0_max, i2_Va0_max = np.unravel_index(np.argmax(Va0), Va0.shape)
    print(f"Va0_max: {Va0_max:.2f} at ({i1_Va0_max}, {i2_Va0_max})")
    dx1, dx2, dx3 = get_dx(coord, dphi)
    for dt in [0.01, 0.05, 0.1]:
        c1, c2, c3 = get_Courant_number(dx1, dx2, dx3, dt, Va0)
        Counrant_numbers = [c1, c2, c3]
        max_Courant_numbers = np.max(c1), np.max(c2), np.max(c3)
        i_C_max = np.argmax(max_Courant_numbers)
        max_Courant_number = max_Courant_numbers[i_C_max]
        i1_Cmax, i2_Cmax = np.unravel_index(np.argmax(Counrant_numbers[i_C_max]), Counrant_numbers[i_C_max].shape)
        print(f"Courant number max for dt={dt}: {max_Courant_number:.2f} at ({i1_Cmax}, {i2_Cmax}) in x{i_C_max+1} direction")
        fig, axes = plt.subplots(1, 3, figsize=(12, 5))
        fig.suptitle(f"Courant number for dt={dt} s, N3={N3}")
        axes[0].imshow(c1, cmap='jet', origin='lower', vmin=0, vmax=1)
        axes[0].set_title('Courant number (dx1)')
        axes[0].set_xlabel('i2')
        axes[0].set_ylabel('i1+1/2')
        axes[1].imshow(c2, cmap='jet', origin='lower', vmin=0, vmax=1)
        axes[1].set_title('Courant number (dx2)')
        axes[1].set_xlabel('i2+1/2')
        axes[1].set_ylabel('i1')
        cax = axes[2].imshow(c3, cmap='jet', origin='lower', vmin=0, vmax=1)
        axes[2].set_title('Courant number (dx3)')
        axes[2].set_xlabel('i2')
        axes[2].set_ylabel('i1')
        plt.colorbar(cax, ax=axes[2])
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        fig.suptitle(f"N1={N1}, N2={N2}, N3={N3}, dt={dt} s, Va0_max={Va0_max*1e-3:.2f} km/s")
        Counrant_number_x2 = Va0_max * dt / dx2
        grid = extend_half_integer_grid(coord, axis=1)
        pcm = axes[0].pcolormesh(grid[0], grid[1], Counrant_number_x2, cmap='jet', shading='auto', vmin=0, vmax=1)
        axes[0].set_title('Courant number (dx2); v = Va_max')
        axes[0].set_aspect('equal')
        # Counrant_number_x3 = Va0_max * dt / dx3
        # grid = extend_half_integer_grid(coord, axis=2)
        # pcm = axes[0].pcolormesh(grid[0], grid[1], Counrant_number_x3, cmap='jet', shading='auto', vmin=0, vmax=1)
        # axes[0].set_title('Courant number (dx3); v = Va_max')
        # axes[0].set_aspect('equal')
        cbar_ax = fig.add_axes([0.06, 0.15, 0.02, 0.7])  # Adjusted position for colorbar
        cbar = plt.colorbar(pcm, cax=cbar_ax)
        cbar.ax.yaxis.set_label_position('left')
        cbar.set_label('Courant number', rotation=90)
        norm = mcolors.LogNorm(vmin=1, vmax=1e3)
        grid = extend_half_integer_grid(coord, axis=2)
        axes[1].pcolormesh(grid[0], grid[1], Rho0, cmap='jet', shading='auto', norm=norm)
        axes[1].set_title('Initial density')
        axes[1].set_aspect('equal')
        pcm = axes[2].pcolormesh(grid[0], grid[1], get_allowed_minimum_density(Va0_max, B0), cmap='jet', shading='auto', norm=norm)
        axes[2].set_title('Allowed minimum density')
        axes[2].set_aspect('equal')
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjusted position for colorbar
        plt.colorbar(pcm, cax=cbar_ax, label='[/cc]')
        plt.show()
        