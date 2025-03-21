import os

import numpy as np
import matplotlib.pyplot as plt

from coordinate import VectorTransformer
from reader import DataReader


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
        # coefficients to convert units of GEMSIS-RC's outputs (SI)
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

    def __getattribute__(self, name):
        if name in self.name_list:
            if not self.is_read[name]:
                print(f"Reading {name} data...")
                DataReader(self, name)
                self.is_read[name] = True
            return self.__dict__[name]
        else:
            raise AttributeError(f"'Run' object has no attribute '{name}'")

    def read_parameters(self, filename='parameter.dat'):
        path_of_file = os.path.join(self.prefix, 'parameter.dat')
        with open(path_of_file, 'rb') as f:
            # domain decompoition
            self.domain = np.fromfile(f, np.int32, 3)[::-1]
            # number of global grid
            self.N1, self.N2, self.N3 = np.fromfile(f, np.int32, 3)
            self.Nm, self.Nv = np.fromfile(f, np.int32, 2)
            # time step
            self.delt = np.fromfile(f, np.float64, 1)
            self.ifdiag, self.ivdiag = np.fromfile(f, np.int32, 2)
            # physical constants
            self.Me, self.Re, self.Qp, self.Mp, self.mu0 = np.fromfile(f, np.float64, 5)
            # free parameters
            self.Rloss, self.N0, self.H0, self.Z0, self.Beta0 = np.fromfile(f, np.float64, 5)
            # simulation domain
            self.Rmin, self.Rmax, self.Lmin, self.Lmax, self.Pmin, self.Pmax = np.fromfile(f, np.float64, 6)
            # number of species
            self.Ns = np.fromfile(f, np.int32, 1)
            # for each species
            self.Qm, self.Mmin, self.Mmax, self.Vmin, self.Vmax, self.mu, self.vp = np.zeros((7, self.Ns))
            for s in range(self.Ns):
                self.Qm[s], self.Mmin[s], self.Mmax[s], self.Vmin[s], self.Vmax[s] = np.fromfile(f, np.float64, 5)
                self.mu[s] = np.fromfile(f, np.float64, self.Nm)
                self.vp[s] = np.fromfile(f, np.float64, self.Nv)
        # array dimension
        self.dims3d = self.N3, self.N2, self.N1
        self.dims5d = self.N3, self.N2, self.N1, self.Nm, self.Nv
        
        

def main():
    pass

if __name__ == "__main__":
    main()