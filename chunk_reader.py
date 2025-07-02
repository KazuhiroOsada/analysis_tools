import numpy as np


def coord_reader(filename, N1, N2, N3):
    """
    filename = 'coord-{d1}-{d2}-{d3}.dat'
    N1, N2, N3 : size of LOCAL gird
    returns scalar(4)              : a, dx1, dx2, dx3
            x1(N1), x2(N2), x3(N3) : logical grid points    
            metric(N3, N2, N1, 3)  : h1, h2, h3
            xyzi(N3, N2, N1, 3)    : cartesian coordinate at cell center
            xyzi(N3, N2, N1, 3)    : cartesian coordinate at cell boundary
    """
    shape = (N3, N2, N1, 3)
    number_of_elements = np.prod(shape)
    with open(filename, 'rb') as f:
        scalar = np.fromfile(f, np.float64, 4)
        x1 = np.fromfile(f, np.float64, N1)
        x2 = np.fromfile(f, np.float64, N2)
        x3 = np.fromfile(f, np.float64, N3)
        metric = np.fromfile(f, np.float64, number_of_elements).reshape(shape)
        xyzi = np.fromfile(f, np.float64, number_of_elements).reshape(shape)
        # cartesian coordinates at cell boundary (different shape)
        xyzh = np.fromfile(f, np.float64, (N3+1)*(N2+1)*(N1+1)*3).reshape((N3+1, N2+1, N1+1, 3))
    return scalar, x1, x2, x3, metric, xyzi, xyzh

def bg_reader(filename, N1, N2, N3):
    """
    filename = 'bg-{d1}-{d2}-{d3}.dat'
    N1, N2, N3 : size of LOCAL gird
    returns B0(N3, N2, N1, 3) : background magnetic field [T]
            Rho0(N3, N2, N1) : initial background density [/m^3]
    """
    shape = (N3, N2, N1)
    number_of_elements = np.prod(shape)
    with open(filename, 'rb') as f:
        B0 = np.fromfile(f, np.float64, number_of_elements * 3).reshape(shape + (3,))
        Rho0 = np.fromfile(f, np.float64, number_of_elements).reshape(shape)
    return B0, Rho0

def field_reader(filename, N1, N2, N3, trange):
    """
    filename = 'field-{d1}-{d2}-{d3}.dat'
    N1, N2, N3 : size of LOCAL gird
    returns V(N3, N2, N1, 3, Nt) : magnetic field [T]
            B(N3, N2, N1, 3, Nt) : velocity [m/s]
    """
    tstep = range(*trange)
    Nt = len(tstep)
    shape = (N3, N2, N1, 3)
    number_of_elements = np.prod(shape)
    # magnetic field and electric drift
    V = np.zeros(shape + (Nt,))
    B = np.zeros(shape + (Nt,))
    bytes_to_skip = 2*number_of_elements*np.dtype(np.float64).itemsize*(trange[2]-1)
    with open(filename, 'rb') as f:
        # seek to the position of the first time step
        f.seek(2*number_of_elements*np.dtype(np.float64).itemsize*trange[0])
        for it in range(Nt):
            V[..., it] = np.fromfile(f, np.float64, number_of_elements).reshape(shape)
            B[..., it] = np.fromfile(f, np.float64, number_of_elements).reshape(shape)
            f.seek(bytes_to_skip, 1)
    return V, B

def current_reader(filename, N1, N2, N3, trange):
    """
    filename = 'current-{d1}-{d2}-{d3}.dat'
    N1, N2, N3 : size of LOCAL gird
    returns Jd(N3, N2, N1, 3, Nt) : drift current [A/m^2]
            Jm(N3, N2, N1, 3, Nt) : gyration current [A/m^2]
            Je(N3, N2, N1, 3, Nt) : electron current [A/m^2]
            Jp(N3, N2, N1, 3, Nt) : polarization current [A/m^2]
    """
    tstep = range(*trange)
    Nt = len(tstep)
    shape = (N3, N2, N1, 3)
    number_of_elements = np.prod(shape)
    # 4 types of current density
    jtypes = 4
    Jd = np.zeros(shape + (Nt,))
    Jm = np.zeros(shape + (Nt,))
    Je = np.zeros(shape + (Nt,))
    Jp = np.zeros(shape + (Nt,))
    bytes_to_skip = jtypes*number_of_elements*np.dtype(np.float64).itemsize*(trange[2]-1)
    with open(filename, 'rb') as f:
        # seek to the position of the first time step
        f.seek(jtypes*number_of_elements*np.dtype(np.float64).itemsize*trange[0])
        for it in range(Nt):
            Jd[..., it] = np.fromfile(f, np.float64, number_of_elements).reshape(shape)
            Jm[..., it] = np.fromfile(f, np.float64, number_of_elements).reshape(shape)
            Je[..., it] = np.fromfile(f, np.float64, number_of_elements).reshape(shape)
            Jp[..., it] = np.fromfile(f, np.float64, number_of_elements).reshape(shape)
            f.seek(bytes_to_skip, 1)
    return Jd, Jm, Je, Jp

def moment_reader(filename, N1, N2, N3, trange):
    """
    filename = 'moment{s+1}-{d1}-{d2}-{d3}.dat'
    N1, N2, N3 : size of LOCAL gird
    returns Rho(N3, N2, N1, Nt) : density [/m^3]
            Vpa(N3, N2, N1, Nt) : bulk velocity [m/s]
            Ppa(N3, N2, N1, Nt) : parallel pressure [Pa]
            Ppe(N3, N2, N1, Nt) : perpendicular pressure [Pa]
    """
    tstep = range(*trange)
    Nt = len(tstep)
    shape = (N3, N2, N1, 4) # for array 'moments'
    number_of_elements = np.prod(shape)
    mtypes = 4  # Rho, Vpa, Ppa, Ppe
    Rho = np.zeros(shape + (Nt,))  # density [/m^3]
    Vpa = np.zeros(shape + (Nt,))  # parallel velocity [m/s]
    Ppa = np.zeros(shape + (Nt,))  # parallel pressure [Pa]
    Ppe = np.zeros(shape + (Nt,))  # perpendicular pressure [Pa]
    bytes_to_skip = mtypes*number_of_elements*np.dtype(np.float64).itemsize*(trange[2]-1)
    with open(filename, 'rb') as f:
        # seek to the position of the first time step
        f.seek(mtypes*number_of_elements*np.dtype(np.float64).itemsize*trange[0])
        for it in range(Nt):
            moments = np.fromfile(f, np.float64, number_of_elements).reshape(shape)
            Rho[..., it] = moments[..., 0]
            Vpa[..., it] = moments[..., 1]
            Ppa[..., it] = moments[..., 2]
            Ppe[..., it] = moments[..., 3]
            f.seek(bytes_to_skip, 1)
    return Rho, Vpa, Ppa, Ppe

def dist_reader(filename, N1, N2, N3, Nm, Nv, trange):
    """
    filename = 'dist{s+1}-{d1}-{d2}-{d3}.dat'
    N1, N2, N3 : size of LOCAL gird
    Nm, Nv : size of velocity grid (32 x 32)
    returns dist(N3, N2, N1, Nm, Nv, Nt) : phase space density [/m^6]
    """
    tstep = range(*trange)
    Nt = len(tstep)
    shape = (N3, N2, N1, Nm, Nv)
    number_of_elements = np.prod(shape)
    dist = np.zeros(shape + (Nt,))
    bytes_to_skip = number_of_elements*np.dtype(np.float64).itemsize*(trange[2]-1)
    with open(filename, 'rb') as f:
        # seek to the position of the first time step
        f.seek(number_of_elements*np.dtype(np.float64).itemsize*trange[0])
        for it in range(Nt):
            dist[..., it] = np.fromfile(f, np.float64, number_of_elements).reshape(shape)
            f.seek(bytes_to_skip, 1)
    return dist
