import os

import numpy as np

from coordinate import VectorTransformer
from reader import DataReader
from equatorial_reader import EquatorialDataReader


class Run:
    """
    Object contains the information/parameters for a specific run
    ----------------------------
    unit for physical quantities
    ----------------------------
    - magnetic field  [nT]
    - electric field  [mV/m]
    - current density [nA/m^2]
    - number density  [1/cm^3]
    - velocity        [km/s]
    - pressure        [nPa]
    """
    def __init__(self, prefix='.'):
        self.prefix = prefix
        # coefficients to convert units of GEMSIS-RC's outputs (all in SI unit)
        self.unitB = 1.0/1.0e-9 # T to nT
        self.unitE = 1.0/1.0e-3 # V/m to mV/m
        self.unitJ = 1.0/1.0e-9 # A/m^2 to nA/m^2
        self.unitN = 1.0/1.0e+6 # 1/m^3 to 1/cm^3
        self.unitV = 1.0/1.0e+3 # m/s to km/s
        self.unitP = 1.0/1.0e-9 # Pa to nPa
        # parameter.dat in default
        self.read_parameters()
        # when name is refered first, read files and store them
        self.name_list = ['coord', 'bg', 'field', 'current', 'moment', 'dist']
        self.is_read = {name: False for name in self.name_list}
        # coordinate system
        self.is_cartesian = {'field': False, 'current': False}
        # call DataReader object
        self.reader = DataReader(self)
        self.eqreader = EquatorialDataReader(self)

    def read_parameters(self, filename='parameter.dat'):
        path_of_file = os.path.join(self.prefix, filename)
        with open(path_of_file, 'rb') as f:
            # domain decompoition
            self.domain = np.fromfile(f, np.int32, 3)[::-1]
            # number of global grid
            self.N1, self.N2, self.N3 = np.fromfile(f, np.int32, 3)
            self.Nm, self.Nv = np.fromfile(f, np.int32, 2)
            # time step
            self.delt = np.fromfile(f, np.float64, 1)[0]
            self.ifdiag, self.ivdiag = np.fromfile(f, np.int32, 2)
            # physical constants
            self.Me, self.Re, self.Qp, self.Mp, self.mu0 = np.fromfile(f, np.float64, 5)
            # free parameters
            self.Rloss, self.N0, self.H0, self.Z0, self.Beta0 = np.fromfile(f, np.float64, 5)
            # simulation domain
            self.Rmin, self.Rmax, self.Lmin, self.Lmax, self.Pmin, self.Pmax = np.fromfile(f, np.float64, 6)
            # number of species
            self.Ns = np.fromfile(f, np.int32, 1)[0]
            if self.Ns > 1: # multi species
                self.Qm, self.Mmin, self.Mmax, self.Vmin, self.Vmax = np.zeros((5, self.Ns))
                self.mu, self.vp = np.zeros((self.Ns, self.Nm)), np.zeros((self.Ns, self.Nv))
                for s in range(self.Ns):
                    self.Qm[s], self.Mmin[s], self.Mmax[s], self.Vmin[s], self.Vmax[s] = np.fromfile(f, np.float64, 5)
                    self.mu[s] = np.fromfile(f, np.float64, self.Nm) # eV/T
                    self.vp[s] = np.fromfile(f, np.float64, self.Nv) # m/s
            else: # single species
                self.Qm, self.Mmin, self.Mmax, self.Vmin, self.Vmax = np.fromfile(f, np.float64, 5)
                self.mu = np.fromfile(f, np.float64, self.Nm) # eV/T
                self.vp = np.fromfile(f, np.float64, self.Nv) # m/s
        # domain decomposed array size
        self.N1_local = self.N1 // self.domain[2]
        self.N2_local = self.N2 // self.domain[1]
        self.N3_local = self.N3 // self.domain[0]

    def set_trange(self, trange, target='f'):
        """
        set time range for data extraction
        trange = (begin, end, interval)
        if target is 'f', set trange for field, current and moment
        if target is 'v', set trange for dist
        """
        if target == 'f':
            self.trange = trange
            self.time = np.arange(*trange) * self.delt * self.ifdiag
            self.Nt = len(self.time)
            for name in self.is_read:
                self.is_read[name] = False
        elif target == 'v':
            # for dist data
            self.trange_v = trange
            self.time_v = np.arange(*trange) * self.delt * self.ivdiag
            self.Nt_v = len(self.time_v)
            # for other data
            v2f = self.ivdiag//self.ifdiag
            self.trange = (trange[0]*v2f, (trange[1]-1)*v2f + 1, trange[2]*v2f)
            for name in self.is_read:
                self.is_read[name] = False
        else:
            print('argument target should be f or v')

    def read(self, name):
        if self.is_read[name]:
            print(f'{name} is already read')
        elif name == 'coord':
            self.reader.read_coord()
        elif name == 'bg':
            self.reader.read_bg()
        elif name == 'field':
            self.reader.read_field(self.trange)
        elif name == 'current':
            self.reader.read_current(self.trange)
        elif name == 'moment':
            for s in range(self.Ns):
                self.reader.read_moment(self.trange, s)
        elif name == 'dist':
            for s in range(self.Ns):
                self.reader.read_dist(self.trange_v, s)
        else:
            print(f'No reader for {name}')
        self.is_read[name] = True

    def read_equatorial(self, name):
        if self.is_read[name]:
            print(f'{name} is already read')
        elif name == 'coord':
            self.eqreader.read_coord()
        elif name == 'bg':
            self.eqreader.read_bg()
        elif name == 'field':
            self.eqreader.read_field(self.trange)
        elif name == 'current':
            self.eqreader.read_current(self.trange)
        elif name == 'moment':
            for s in range(self.Ns):
                self.eqreader.read_moment(self.trange, s)
        elif name == 'dist':
            for s in range(self.Ns):
                self.eqreader.read_dist(self.trange_v, s)
        else:
            print(f'No reader for {name}')
        self.is_read[name] = True

    def transform(self, name):
        """
        transform vector data from dipole to cartesian coordinate
        """
        transformer = VectorTransformer(self.Xi, self.Yi, self.Zi)
        if name == 'field':
            self.B0 = transformer(self.B0)
            for it in range(self.Nt):
                self.B[..., it] = transformer(self.B[..., it])
                self.V[..., it] = transformer(self.V[..., it])
            self.is_cartesian['field'] = True
        elif name == 'current':
            for it in range(self.Nt):
                self.Jd[..., it] = transformer(self.Jd[..., it])
                self.Jm[..., it] = transformer(self.Jm[..., it])
                self.Je[..., it] = transformer(self.Je[..., it])
                self.Jp[..., it] = transformer(self.Jp[..., it])
                self.Jtot[..., it] = transformer(self.Jtot[..., it])
            self.is_cartesian['current'] = True
        else:
            print(f'No transformation for {name}')

    def calc_electric_field(self):
        """
        calculate electric field from V and B
        E = - V x B
        """
        Btot = self.B + self.B0[..., None]
        E = - np.cross(self.V, Btot, axis=-2)
        self.E = E * self.unitE / self.unitV / self.unitB
