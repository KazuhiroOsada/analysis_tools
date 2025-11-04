import numpy as np


def coord_reader(filename, N1_local, N2_local, N3_local):
    """
    Parameters
    ----------
    filename                     : 'coord-{d1}-{d2}-{d3}.dat'
    N1_local, N2_local, N3_local : size of LOCAL gird

    Returns
    -------
    scalar             : a, dx1, dx2, dx3
    x1, x2, x3         : 1D arrays of logical grid points, shape=(N1_local,), (N2_local,), (N3_local,)
    metric, xyzi, xyzh : h1, h2, h3 and cartesian coordinates at cell center and boundary
                         shape=(N3_local, N2_local, N1_local, 3)
    """
    shape = (N3_local, N2_local, N1_local, 3)
    number_of_elements = np.prod(shape)
    with open(filename, 'rb') as f:
        scalar = np.fromfile(f, np.float64, 4)
        x1 = np.fromfile(f, np.float64, N1_local)
        x2 = np.fromfile(f, np.float64, N2_local)
        x3 = np.fromfile(f, np.float64, N3_local)
        metric = np.fromfile(f, np.float64, number_of_elements).reshape(shape)
        xyzi = np.fromfile(f, np.float64, number_of_elements).reshape(shape)
        # cartesian coordinates at cell boundary (different shape)
        xyzh = np.fromfile(f, np.float64, (N3_local+1)*(N2_local+1)*(N1_local+1)*3).reshape((N3_local+1, N2_local+1, N1_local+1, 3))
    return scalar, x1, x2, x3, metric, xyzi, xyzh

def bg_reader(filename, N1_local, N2_local, N3_local):
    """
    Parameters
    ----------
    filename                     : 'bg-{d1}-{d2}-{d3}.dat'
    N1_local, N2_local, N3_local : size of LOCAL gird

    Returns
    -------
    B0, Rho0 : background magnetic field [T], initial background density [/m^3]
               shape=(N3_local, N2_local, N1_local, 3), (N3_local, N2_local, N1_local)
    """
    shape = (N3_local, N2_local, N1_local)
    number_of_elements = np.prod(shape)
    with open(filename, 'rb') as f:
        B0 = np.fromfile(f, np.float64, number_of_elements * 3).reshape(shape + (3,))
        Rho0 = np.fromfile(f, np.float64, number_of_elements).reshape(shape)
    return B0, Rho0

def field_reader(filename, N1_local, N2_local, N3_local, trange):
    """
    Parameters
    ----------
    filename                     : 'field-{d1}-{d2}-{d3}.dat'
    N1_local, N2_local, N3_local : size of LOCAL gird

    Returns
    -------
    V, B : velocity [m/s], magnetic field [T], shape=(N3_local, N2_local, N1_local, 3, Nt)
    """
    tstep = range(*trange)
    Nt = len(tstep)
    shape = (N3_local, N2_local, N1_local, 3)
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

def current_reader(filename, N1_local, N2_local, N3_local, trange):
    """
    Parameters
    ----------
    filename                     : 'current-{d1}-{d2}-{d3}.dat'
    N1_local, N2_local, N3_local : size of LOCAL gird

    Returns
    -------
    Jd, Jm, Je, Jp : drift, gyration, electron, polarization current density [A/m^2]
                     shape=(N3_local, N2_local, N1_local, 3, Nt)
    """
    tstep = range(*trange)
    Nt = len(tstep)
    shape = (N3_local, N2_local, N1_local, 3)
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

def moment_reader(filename, N1_local, N2_local, N3_local, trange):
    """
    Parameters
    ----------
    filename                     : 'moment{s+1}-{d1}-{d2}-{d3}.dat'
    N1_local, N2_local, N3_local : size of LOCAL gird

    Returns
    -------
    Rho, Vpa, Ppa, Ppe : density [/m^3], parallel bulk velocity [m/s],
                         parallel pressure [Pa], perpendicular pressure [Pa]
                         shape=(N3_local, N2_local, N1_local, Nt)
    """
    tstep = range(*trange)
    Nt = len(tstep)
    shape = (N3_local, N2_local, N1_local, 4) # for array 'moments'
    number_of_elements = np.prod(shape)
    Rho = np.zeros(shape[:-1] + (Nt,))  # density [/m^3]
    Vpa = np.zeros(shape[:-1] + (Nt,))  # parallel velocity [m/s]
    Ppa = np.zeros(shape[:-1] + (Nt,))  # parallel pressure [Pa]
    Ppe = np.zeros(shape[:-1] + (Nt,))  # perpendicular pressure [Pa]
    bytes_to_skip = number_of_elements*np.dtype(np.float64).itemsize*(trange[2]-1)
    with open(filename, 'rb') as f:
        # seek to the position of the first time step
        f.seek(number_of_elements*np.dtype(np.float64).itemsize*trange[0])
        for it in range(Nt):
            moments = np.fromfile(f, np.float64, number_of_elements).reshape(shape)
            Rho[..., it] = moments[..., 0]
            Vpa[..., it] = moments[..., 1]
            Ppa[..., it] = moments[..., 2]
            Ppe[..., it] = moments[..., 3]
            f.seek(bytes_to_skip, 1)
    return Rho, Vpa, Ppa, Ppe

def dist_reader(filename, N1_local, N2_local, N3_local, Nm, Nv, trange):
    """
    Parameters
    ----------
    filename                     : 'dist{s+1}-{d1}-{d2}-{d3}.dat'
    N1_local, N2_local, N3_local : size of LOCAL gird
    Nm, Nv                       : size of velocity grid (32 x 32)

    Returns
    -------
    dist : phase space density [/m^6], shape=(N3_local, N2_local, N1_local, Nm, Nv, Nt)
    """
    tstep = range(*trange)
    Nt = len(tstep)
    shape = (N3_local, N2_local, N1_local, Nm, Nv)
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
