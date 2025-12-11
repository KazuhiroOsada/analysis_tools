import os
from pathlib import Path
from enum import Enum

import numpy as np

from coordinate import VectorTransformer
from reader import DataReader
from equatorial_reader import EquatorialDataReader


class Run:
    """
    This object reads and contains the information, parameters and data for a specific run,
    and also provides some simple data processing methods
    For details of data reading, see DataReader class in reader.py and EquatorialDataReader class in equatorial_reader.py
    Also, check chunk_reader.py for domain decomposed data reading functions

    Parameters
    ----------
    prefix : directory path where data files are stored (parameter.dat is required)

    Example
    -------
    >>> run = Run('run_directory')
    >>> run.read('coord')
    >>> run.Xi.shape # (N3, N2, N1)
    >>> run.set_trange((0, 200, 1))
    >>> run.read('field')
    >>> run.B.shape # (N3, N2, N1, 3, Nt)

    Unit for physical quantities
    ----------------------------
    - magnetic field  [nT]
    - electric field  [mV/m]
    - current density [nA/m^2]
    - number density  [1/cm^3]
    - velocity        [km/s]
    - pressure        [nPa]
    """
    class ReadState(Enum):
        NONE       = 0
        ALL        = 1
        EQUATORIAL = 2

    def __init__(self, prefix):
        self.prefix = Path(prefix)
        # coefficients to convert units of GEMSIS-RC's outputs (all in SI unit)
        self.unitB = 1.0/1.0e-9 # T to nT
        self.unitE = 1.0/1.0e-3 # V/m to mV/m
        self.unitJ = 1.0/1.0e-9 # A/m^2 to nA/m^2
        self.unitN = 1.0/1.0e+6 # 1/m^3 to 1/cm^3
        self.unitV = 1.0/1.0e+3 # m/s to km/s
        self.unitP = 1.0/1.0e-9 # Pa to nPa
        self.read_parameters()
        self.name_list = ['coord', 'bg', 'field', 'current', 'moment', 'dist']
        self.is_read = {name: Run.ReadState.NONE for name in self.name_list}
        # coordinate system
        self.is_cartesian = {'field': False, 'current': False}
        # call DataReader object
        self.reader = DataReader(self)
        self.eqreader = EquatorialDataReader(self)

    def read_parameters(self):
        path_of_file = os.path.join(self.prefix, 'parameter.dat')
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
        Set time range for data extraction
        
        Parameters
        ----------
        trange : tuple of (begin, end, interval)
        target : 'f' for field/current/moment data, 'v' for dist data
        """
        if target not in ['f', 'v']:
            raise ValueError(f"Invalid target: {target}. Must be 'f' or 'v'.")
        
        if target == 'f':
            self.trange = trange
            self.time = np.arange(*trange) * self.delt * self.ifdiag
            self.Nt = len(self.time)
        elif target == 'v':
            # for dist data
            self.trange_v = trange
            self.time_v = np.arange(*trange) * self.delt * self.ivdiag
            self.Nt_v = len(self.time_v)
            # for other data
            v2f = self.ivdiag//self.ifdiag
            self.trange = (trange[0]*v2f, (trange[1]-1)*v2f + 1, trange[2]*v2f)
        
        # reset is_read status
        for name in self.is_read:
            if name == 'bg' or name == 'coord':
                continue
            elif self.is_read[name] != Run.ReadState.NONE:
                print(f'Notice: trange is changed after reading {name} data')
                self.is_read[name] = Run.ReadState.NONE

    def read(self, name):
        """
        Read data files and store them into the Run object

        Parameters
        ----------
        name : 'coord', 'bg', 'field', 'current', 'moment', or 'dist'
        """
        if name not in self.name_list:
            raise ValueError(f'Invalid name: {name}')
        
        if self.is_read[name] == Run.ReadState.ALL:
            print(f'{name} is already read')
            return
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
    
        if self.is_read[name] == Run.ReadState.EQUATORIAL:
            print(f'Notice: {name} data is overwritten by full 3D data')
        self.is_read[name] = Run.ReadState.ALL

    def read_equatorial(self, name):
        """
        Read data files on the equatorial plane and store them into the Run object

        Parameters
        ----------
        name : 'coord', 'bg', 'field', 'current', 'moment', or 'dist'
        """
        if name not in self.name_list:
            raise ValueError(f'Invalid name: {name}')

        if self.is_read[name] == Run.ReadState.EQUATORIAL:
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

        if self.is_read[name] == Run.ReadState.ALL:
            print(f'Notice: {name} data is overwritten by equatorial data')
        self.is_read[name] = Run.ReadState.EQUATORIAL

    def transform(self, name):
        """
        Transform vector data from dipole to cartesian coordinate

        Parameters
        ----------
        name : 'field' or 'current', vectors to be transformed
        """
        if name not in ['field', 'current']:
            raise ValueError(f"Invalid name: {name}. Must be 'field' or 'current'.")

        if self.is_cartesian[name]:
            print(f'{name} data is already in cartesian coordinate')
            return
        
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

    def calc_electric_field(self):
        """
        Calculate electric field from V and B (E = - V x B)
        Results are stored in self.E [mV/m] (shape: same as self.B and self.V)
        """
        Btot = self.B + self.B0[..., None]
        E = - np.cross(self.V, Btot, axis=-2)
        self.E = E * self.unitE / self.unitV / self.unitB

    def calc_magnetic_amplitude(self):
        """
        Calculate magnetic field amplitude
        Results are stored in self.Babs [nT] (shape: (N3, N2, N1, Nt) or (N3, N2, Nt))
        """
        Btot = self.B + self.B0[..., None]
        self.Babs = np.linalg.norm(Btot, axis=-2)

    def resolve_global_idx(self, i1, i2, i3):
        """
        Parameters
        ----------
        i1, i2, i3 : global indices for x1, x2, x3 directions

        Returns
        -------
        d1, d2, d3 : domain indices for x1, x2, x3 directions
        l1, l2, l3 : local indices for x1, x2, x3 directions
        """
        d1, l1 = i1 // self.N1_local, i1 % self.N1_local
        d2, l2 = i2 // self.N2_local, i2 % self.N2_local
        d3, l3 = i3 // self.N3_local, i3 % self.N3_local
        return d1, d2, d3, l1, l2, l3
