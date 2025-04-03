import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np


# number of threads for parallel processing if max_workers is None, it will be set automatically
max_workers = None


class EquatorialDataReader:
    """
    Reader for domain decomposed data files of GEMSIS-RC
    This class reads data files only on the equatorial plane (i1=N1//2)
    """
    def __init__(self, run, i1=None):
        self.run = run
        self.domain = run.domain
        self.Nd3, self.Nd2, self.Nd1 = run.domain # number of domain decomposition
        self.N3, self.N2, self.N1, self.Nm, self.Nv = run.N3, run.N2, run.N1, run.Nm, run.Nv
        self.N3_local, self.N2_local, self.N1_local = run.N3//self.domain[0], run.N2//self.domain[1], run.N1//self.domain[2]
        self.ext = '.dat'
        # find domain and local index for i1
        if i1 is None:
            self.i1 = self.N1//2
        elif i1 < 0 or i1 >= self.N1:
            raise ValueError(f'i1={i1} is out of range [0, {self.N1})')
        self.d1 = self.i1 // self.N1_local
        self.l1 = self.i1 % self.N1_local

    def get_file_path(self, name, d2, d3, s=None):
        file_path = os.path.join(self.run.prefix, name + '-{:02d}-{:02d}-{:02d}'.format(self.d1, d2, d3) + self.ext)
        if s is not None:
            file_path = os.path.join(self.run.prefix, name + '{:d}-{:02d}-{:02d}-{:02d}'.format(s+1, self.d1, d2, d3) + self.ext)
        return file_path
    
    def thread_parallel_processing(self, process_chunk):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_chunk, d2, d3)
                for d3 in range(self.Nd3)
                for d2 in range(self.Nd2)
            ]
            for future in futures:
                future.result()

    def read_coord(self):
        """
        read 'coord-**-**-**.dat' files for coordinate data and parameters on the equatorial plane
        """
        shape_global = (self.N3, self.N2, 3)
        shape_local = (self.N3_local, self.N2_local, self.N1_local, 3)
        elements_local = self.N3_local*self.N2_local*self.N1_local*3  
        # scalar and logical grid points
        scalar = np.zeros(4) # a, dx1, dx2, dx3
        x1, x2, x3 = np.zeros(1), np.zeros(self.N2), np.zeros(self.N3)
        # metric and cartesian coordinate at cell center
        metric = np.zeros(shape_global)
        xyzi = np.zeros(shape_global)

        def process_chunk(d2, d3):
            file_path = self.get_file_path('coord', d2, d3)
            with open(file_path, 'rb') as f:
                # read scalar
                scalar[:] = np.fromfile(f, np.float64, 4)
                i2_local = np.arange(self.N2_local) + self.N2_local*d2
                i3_local = np.arange(self.N3_local) + self.N3_local*d3
                # read logical grid points
                x1 = np.fromfile(f, np.float64, self.N1_local)[self.l1]
                x2[i2_local] = np.fromfile(f, np.float64, self.N2_local)
                x3[i3_local] = np.fromfile(f, np.float64, self.N3_local)
                # read metric and cartesian coordinate at cell center
                idx = (i3_local[:, None], i2_local[None, :], slice(None))                
                metric[idx] = np.fromfile(f, np.float64, elements_local).reshape(shape_local)[:,:,self.l1,:]
                xyzi[idx] = np.fromfile(f, np.float64, elements_local).reshape(shape_local)[:,:,self.l1,:]
        
        self.thread_parallel_processing(process_chunk)

        # pack data into run object
        self.run.a, self.run.dx1, self.run.dx2, self.run.dx3 = scalar
        self.run.b = np.arcsinh(self.run.a)
        self.run.x1, self.run.x2, self.run.x3 = x1, x2, x3
        self.run.h1, self.run.h2, self.run.h3 = metric[..., 0], metric[..., 1], metric[..., 2]
        # unit is Re for cartesian coordinate
        self.run.Xi, self.run.Yi, self.run.Zi = xyzi[..., 0]/self.run.Re, xyzi[..., 1]/self.run.Re, xyzi[..., 2]/self.run.Re
        # half integer grid
        self.run.xh1 = self.run.x1 + 0.5*self.run.dx1
        self.run.xh2 = self.run.x2 + 0.5*self.run.dx2
        self.run.xh3 = self.run.x3 + 0.5*self.run.dx3

    def read_bg(self):
        """
        read 'bg-**-**-**.dat' files for background magnetic field and density on the equatorial plane
        """
        shape_global_B0 = (self.N3, self.N2, 3)
        shape_local_B0 = (self.N3_local, self.N2_local, self.N1_local, 3)
        elements_local_B0 = self.N3_local*self.N2_local*self.N1_local*3
        shape_global_Rho0 = (self.N3, self.N2)
        shape_local_Rho0 = (self.N3_local, self.N2_local, self.N1_local)
        elements_local_Rho0 = self.N3_local*self.N2_local*self.N1_local
        # background magnetic field and density
        B0 = np.zeros(shape_global_B0)
        Rho0 = np.zeros(shape_global_Rho0)

        def process_chunk(d2, d3):
            file_path = self.get_file_path('bg', d2, d3)
            with open(file_path, 'rb') as f:
                i2_local = np.arange(self.N2_local) + self.N2_local*d2
                i3_local = np.arange(self.N3_local) + self.N3_local*d3
                # read background magnetic field
                idx = (i3_local[:, None], i2_local[None, :], slice(None))
                B0[idx] = np.fromfile(f, np.float64, elements_local_B0).reshape(shape_local_B0)[:, :, self.l1, :]
                # read background density
                Rho0[idx[:-1]] = np.fromfile(f, np.float64, elements_local_Rho0).reshape(shape_local_Rho0)[:, :, self.l1]

        self.thread_parallel_processing(process_chunk)

        # pack data into run object
        self.run.B0 = B0 * self.run.unitB
        self.run.Rho0 = Rho0 * self.run.unitN

    def read_field(self, trange):
        """
        read 'field-**-**-**.dat' files for magnetic field and electric drift on the equatorial plane
        """
        tstep = range(*trange)
        Nt = len(tstep)
        shape_global = (self.N3, self.N2, 3, Nt)
        shape_local = (self.N3_local, self.N2_local, self.N1_local, 3)
        elements_local = self.N3_local*self.N2_local*self.N1_local*3
        # magnetic field and electric drift
        V = np.zeros(shape_global)
        B = np.zeros(shape_global)

        def process_chunk(d2, d3):
            file_path = self.get_file_path('field', d2, d3)
            with open(file_path, 'rb') as f:
                i2_local = np.arange(self.N2_local) + self.N2_local*d2
                i3_local = np.arange(self.N3_local) + self.N3_local*d3
                # seek to the position of the first time step
                f.seek(elements_local * np.dtype(np.float64).itemsize * trange[0])
                for it in range(Nt):
                    # read electric drift
                    idx = (i3_local[:, None], i2_local[None, :], slice(None), it)
                    V[idx] = np.fromfile(f, np.float64, elements_local).reshape(shape_local)[:, :, self.l1, :]
                    # read magnetic field
                    B[idx] = np.fromfile(f, np.float64, elements_local).reshape(shape_local)[:, :, self.l1, :]
                    # seek to the position of the next time step
                    f.seek(2 * elements_local * np.dtype(np.float64).itemsize * (trange[2]-1), 1)

        self.thread_parallel_processing(process_chunk)

        # pack data into run object
        self.run.B = B * self.run.unitB
        self.run.V = V * self.run.unitV

    def read_current(self, trange):
        """
        read 'current-**-**-**.dat' files for current density on the equatorial plane
        """
        tstep = range(*trange)
        Nt = len(tstep)
        shape_global = (self.N3, self.N2, 3, Nt)
        shape_local = (self.N3_local, self.N2_local, self.N1_local, 3)
        elements_local = self.N3_local*self.N2_local*self.N1_local*3
        # 4 types of current density
        jtypes = 4
        Jd = np.zeros(shape_global)
        Jm = np.zeros(shape_global)
        Je = np.zeros(shape_global)
        Jp = np.zeros(shape_global)

        def process_chunk(d2, d3):
            file_path = self.get_file_path('current', d2, d3)
            with open(file_path, 'rb') as f:
                i2_local = np.arange(self.N2_local) + self.N2_local*d2
                i3_local = np.arange(self.N3_local) + self.N3_local*d3
                # seek to the position of the first time step
                f.seek(elements_local * jtypes * np.dtype(np.float64).itemsize * trange[0])
                for it in range(Nt):
                    # read current density
                    idx = (i3_local[:, None], i2_local[None, :], slice(None), it)
                    Jd[idx] = np.fromfile(f, np.float64, elements_local).reshape(shape_local)[:, :, self.l1, :]
                    Jm[idx] = np.fromfile(f, np.float64, elements_local).reshape(shape_local)[:, :, self.l1, :]
                    Je[idx] = np.fromfile(f, np.float64, elements_local).reshape(shape_local)[:, :, self.l1, :]
                    Jp[idx] = np.fromfile(f, np.float64, elements_local).reshape(shape_local)[:, :, self.l1, :]
                    # seek to the position of the next time step
                    f.seek(elements_local * jtypes * np.dtype(np.float64).itemsize * (trange[2]-1), 1)

        self.thread_parallel_processing(process_chunk)

        # pack data into run object
        self.run.Jd = Jd * self.run.unitJ
        self.run.Jm = Jm * self.run.unitJ
        self.run.Je = Je * self.run.unitJ
        self.run.Jp = Jp * self.run.unitJ
        self.run.Jtot = (Jd + Jm + Je + Jp) * self.run.unitJ

    def read_moment(self, trange, s=0):
        """
        read 'moment{s+1}-**-**-**.dat' files for moments on the equatorial plane
        """
        tstep = range(*trange)
        Nt = len(tstep)
        shape_global = (self.N3, self.N2, Nt)  # for each moment values
        shape_local = (self.N3_local, self.N2_local, self.N1_local, 4)  # for moments
        elements_local = self.N3_local*self.N2_local*self.N1_local*4  # for moments
        # moments
        Rho = np.zeros(shape_global)
        Vpa = np.zeros(shape_global)
        Ppa = np.zeros(shape_global)
        Ppe = np.zeros(shape_global)

        def process_chunk(d2, d3):
            file_path = self.get_file_path('moment', d2, d3, s)
            with open(file_path, 'rb') as f:
                i2_local = np.arange(self.N2_local) + self.N2_local*d2
                i3_local = np.arange(self.N3_local) + self.N3_local*d3
                # seek to the position of the first time step
                f.seek(elements_local * np.dtype(np.float64).itemsize * trange[0])
                for it in range(Nt):
                    # read moments
                    moments = np.fromfile(f, np.float64, elements_local).reshape(shape_local)
                    idx = (i3_local[:, None], i2_local[None, :], it)
                    Rho[idx] = moments[:, :, self.l1, 0]
                    Vpa[idx] = moments[:, :, self.l1, 1]
                    Ppa[idx] = moments[:, :, self.l1, 2]
                    Ppe[idx] = moments[:, :, self.l1, 3]
                    # seek to the position of the next time step
                    f.seek(elements_local * np.dtype(np.float64).itemsize * (trange[2]-1), 1)

        self.thread_parallel_processing(process_chunk)

        # pack data into run object
        if self.run.Ns == 1:
            self.run.Rho = Rho * self.run.unitN
            self.run.Vpa = Vpa * self.run.unitV
            self.run.Ppa = Ppa * self.run.unitP
            self.run.Ppe = Ppe * self.run.unitP
        else:
            if not hasattr(self.run, 'Rho'):
                self.run.Rho = [[] for _ in range(self.run.Ns)]
                self.run.Vpa = [[] for _ in range(self.run.Ns)]
                self.run.Ppa = [[] for _ in range(self.run.Ns)]
                self.run.Ppe = [[] for _ in range(self.run.Ns)]
            self.run.Rho[s] = Rho * self.run.unitN
            self.run.Vpa[s] = Vpa * self.run.unitV
            self.run.Ppa[s] = Ppa * self.run.unitP
            self.run.Ppe[s] = Ppe * self.run.unitP

    def read_dist(self, trange, s=0):
        """
        read 'dist{s+1}-**-**-**.dat' files for phase space density on the equatorial plane
        """
        tstep = range(*trange)
        Nt = len(tstep)
        shape_global = (self.N3, self.N2, self.Nm, self.Nv, Nt)
        shape_local = (self.N3_local, self.N2_local, self.N1_local, self.Nm, self.Nv)
        elements_local = self.N3_local*self.N2_local*self.N1_local*self.Nm*self.Nv
        # phase space density
        dist = np.zeros(shape_global)

        def process_chunk(d2, d3):
            file_path = self.get_file_path('dist', d2, d3, s)
            with open(file_path, 'rb') as f:
                i2_local = np.arange(self.N2_local) + self.N2_local*d2
                i3_local = np.arange(self.N3_local) + self.N3_local*d3
                # seek to the position of the first time step
                f.seek(elements_local * np.dtype(np.float64).itemsize * trange[0])
                for it in range(Nt):
                    # read phase space density
                    idx = (i3_local[:, None], i2_local[None, :], slice(None), slice(None), it)
                    dist[idx] = np.fromfile(f, np.float64, elements_local).reshape(shape_local)[:, :, self.l1, :, :]
                    # seek to the position of the next time step
                    f.seek(elements_local * np.dtype(np.float64).itemsize * (trange[2]-1), 1)

        self.thread_parallel_processing(process_chunk)

        # pack data into run object
        if self.run.Ns == 1:
            self.run.dist = dist
        else:
            if not hasattr(self.run, 'dist'):
                self.run.dist = [[] for _ in range(self.run.Ns)]
            self.run.dist[s] = dist
