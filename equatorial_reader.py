import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from chunk_reader import coord_reader, bg_reader, field_reader, current_reader, moment_reader, dist_reader


# number of threads for parallel processing if max_workers is None, it will be set automatically
max_workers = None


class EquatorialDataReader:
    """
    Reader for domain decomposed data files of GEMSIS-RC
    This class is used internally in the Run class in base.py and stores data into the Run object
    This class is designed to read data on the equatorial plane (i1 = N1//2) only in order to reduce memory usage

    Parameters
    ----------
    run : Run object
    i1  : index in x1 direction (default: N1//2)
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
        Read 'coord-**-**-**.dat' files for coordinate data and parameters on the equatorial plane
        """
        shape_global = (self.N3, self.N2, self.N1_local,3)
        # scalar and logical grid points
        scalar = np.zeros(4) # a, dx1, dx2, dx3
        x1, x2, x3 = np.zeros(self.N1_local), np.zeros(self.N2), np.zeros(self.N3)
        # metric and cartesian coordinate at cell center
        metric = np.zeros(shape_global)
        xyzi = np.zeros(shape_global)
        # cartesian coordinate at cell boundary (with different local shape !)
        xyzh = np.zeros((self.N3+1, self.N2+1, self.N1_local+1, 3))

        def process_chunk(d2, d3):
            file_path = self.get_file_path('coord', d2, d3)
            i2_local = np.arange(self.N2_local) + self.N2_local*d2
            i3_local = np.arange(self.N3_local) + self.N3_local*d3
            idx = (i3_local[:, None], i2_local[None, :], slice(None), slice(None))
            h2_local = np.arange(self.N2_local+1) + self.N2_local*d2
            h3_local = np.arange(self.N3_local+1) + self.N3_local*d3
            hidx = (h3_local[:, None], h2_local[None, :], slice(None), slice(None))
            scalar[:], x1[:], x2[i2_local], x3[i3_local], \
            metric[idx], xyzi[idx], xyzh[hidx] = coord_reader(file_path, self.N1_local, self.N2_local, self.N3_local)

        self.thread_parallel_processing(process_chunk)

        # pack data into run object
        self.run.a, self.run.dx1, self.run.dx2, self.run.dx3 = scalar
        self.run.b = np.arcsinh(self.run.a)
        self.run.x1, self.run.x2, self.run.x3 = x1[self.l1], x2, x3
        self.run.h1, self.run.h2, self.run.h3 = metric[..., self.l1, 0], metric[..., self.l1, 1], metric[..., self.l1, 2]
        # unit is Re for cartesian coordinate
        self.run.Xi, self.run.Yi, self.run.Zi = xyzi[..., self.l1, 0]/self.run.Re, xyzi[..., self.l1, 1]/self.run.Re, xyzi[..., self.l1, 2]/self.run.Re
        self.run.Xh, self.run.Yh, self.run.Zh = xyzh[..., self.l1, 0]/self.run.Re, xyzh[..., self.l1, 1]/self.run.Re, xyzh[..., self.l1, 2]/self.run.Re
        # half integer grid
        self.run.xh1 = self.run.x1 + 0.5*self.run.dx1
        self.run.xh2 = self.run.x2 + 0.5*self.run.dx2
        self.run.xh3 = self.run.x3 + 0.5*self.run.dx3

    def read_bg(self):
        """
        Read 'bg-**-**-**.dat' files for background magnetic field and density on the equatorial plane
        """
        shape_global_B0 = (self.N3, self.N2, self.N1_local, 3)
        shape_global_Rho0 = (self.N3, self.N2, self.N1_local)
        # background magnetic field and density
        B0 = np.zeros(shape_global_B0)
        Rho0 = np.zeros(shape_global_Rho0)

        def process_chunk(d2, d3):
            file_path = self.get_file_path('bg', d2, d3)
            i2_local = np.arange(self.N2_local) + self.N2_local*d2
            i3_local = np.arange(self.N3_local) + self.N3_local*d3
            idx_B0 = (i3_local[:, None], i2_local[None, :], slice(None), slice(None))
            idx_Rho0 = (i3_local[:, None], i2_local[None, :], slice(None))
            B0[idx_B0], Rho0[idx_Rho0] = bg_reader(file_path, self.N1_local, self.N2_local, self.N3_local)

        self.thread_parallel_processing(process_chunk)

        # pack data into run object
        self.run.B0 = B0[..., self.l1, :] * self.run.unitB
        self.run.Rho0 = Rho0[..., self.l1] * self.run.unitN

    def read_field(self, trange):
        """
        Read 'field-**-**-**.dat' files for magnetic field and electric drift on the equatorial plane
        """
        tstep = range(*trange)
        Nt = len(tstep)
        shape_global = (self.N3, self.N2, self.N1_local, 3, Nt)
        # magnetic field and electric drift
        V = np.zeros(shape_global)
        B = np.zeros(shape_global)

        def process_chunk(d2, d3):
            file_path = self.get_file_path('field', d2, d3)
            i2_local = np.arange(self.N2_local) + self.N2_local*d2
            i3_local = np.arange(self.N3_local) + self.N3_local*d3
            idx = (i3_local[:, None], i2_local[None, :], slice(None), slice(None), slice(None))
            V[idx], B[idx] = field_reader(file_path, self.N1_local, self.N2_local, self.N3_local, trange)

        self.thread_parallel_processing(process_chunk)

        # pack data into run object
        self.run.B = B[..., self.l1, :, :] * self.run.unitB
        self.run.V = V[..., self.l1, :, :] * self.run.unitV

    def read_current(self, trange):
        """
        Read 'current-**-**-**.dat' files for current density on the equatorial plane
        """
        tstep = range(*trange)
        Nt = len(tstep)
        shape_global = (self.N3, self.N2, self.N1_local, 3, Nt)
        # 4 types of current density
        Jd = np.zeros(shape_global)
        Jm = np.zeros(shape_global)
        Je = np.zeros(shape_global)
        Jp = np.zeros(shape_global)

        def process_chunk(d2, d3):
            file_path = self.get_file_path('current', d2, d3)
            i2_local = np.arange(self.N2_local) + self.N2_local*d2
            i3_local = np.arange(self.N3_local) + self.N3_local*d3
            idx = (i3_local[:, None], i2_local[None, :], slice(None), slice(None), slice(None))
            Jd[idx], Jm[idx], Je[idx], Jp[idx] = current_reader(file_path, self.N1_local, self.N2_local, self.N3_local, trange)

        self.thread_parallel_processing(process_chunk)

        # pack data into run object
        self.run.Jd = Jd[..., self.l1, :, :] * self.run.unitJ
        self.run.Jm = Jm[..., self.l1, :, :] * self.run.unitJ
        self.run.Je = Je[..., self.l1, :, :] * self.run.unitJ
        self.run.Jp = Jp[..., self.l1, :, :] * self.run.unitJ
        self.run.Jtot = self.run.Jd + self.run.Jm + self.run.Je + self.run.Jp

    def read_moment(self, trange, s=0):
        """
        Read 'moment{s+1}-**-**-**.dat' files for moments on the equatorial plane
        """
        tstep = range(*trange)
        Nt = len(tstep)
        shape_global = (self.N3, self.N2, self.N1_local, Nt)  # for each moment values
        # moments
        Rho = np.zeros(shape_global)
        Vpa = np.zeros(shape_global)
        Ppa = np.zeros(shape_global)
        Ppe = np.zeros(shape_global)

        def process_chunk(d2, d3):
            file_path = self.get_file_path('moment', d2, d3, s)
            i2_local = np.arange(self.N2_local) + self.N2_local*d2
            i3_local = np.arange(self.N3_local) + self.N3_local*d3
            idx = (i3_local[:, None], i2_local[None, :], slice(None), slice(None))
            Rho[idx], Vpa[idx], Ppa[idx], Ppe[idx] = moment_reader(file_path, self.N1_local, self.N2_local, self.N3_local, trange)

        self.thread_parallel_processing(process_chunk)

        # pack data into run object
        if self.run.Ns == 1:
            self.run.Rho = Rho[..., self.l1, :] * self.run.unitN
            self.run.Vpa = Vpa[..., self.l1, :] * self.run.unitV
            self.run.Ppa = Ppa[..., self.l1, :] * self.run.unitP
            self.run.Ppe = Ppe[..., self.l1, :] * self.run.unitP
        else:
            if not hasattr(self.run, 'Rho'):
                self.run.Rho = [[] for _ in range(self.run.Ns)]
                self.run.Vpa = [[] for _ in range(self.run.Ns)]
                self.run.Ppa = [[] for _ in range(self.run.Ns)]
                self.run.Ppe = [[] for _ in range(self.run.Ns)]
            self.run.Rho[s] = Rho[..., self.l1, :] * self.run.unitN
            self.run.Vpa[s] = Vpa[..., self.l1, :] * self.run.unitV
            self.run.Ppa[s] = Ppa[..., self.l1, :] * self.run.unitP
            self.run.Ppe[s] = Ppe[..., self.l1, :] * self.run.unitP

    def read_dist(self, trange, s=0):
        """
        Read 'dist{s+1}-**-**-**.dat' files for phase space density on the equatorial plane
        """
        tstep = range(*trange)
        Nt = len(tstep)
        shape_global = (self.N3, self.N2, self.N1_local, self.Nm, self.Nv, Nt)
        # phase space density
        dist = np.zeros(shape_global)

        def process_chunk(d2, d3):
            file_path = self.get_file_path('dist', d2, d3, s)
            i2_local = np.arange(self.N2_local) + self.N2_local*d2
            i3_local = np.arange(self.N3_local) + self.N3_local*d3
            idx = (i3_local[:, None], i2_local[None, :], slice(None), slice(None), slice(None), slice(None))
            dist[idx] = dist_reader(file_path, self.N1_local, self.N2_local, self.N3_local, self.Nm, self.Nv, trange)

        self.thread_parallel_processing(process_chunk)

        # pack data into run object
        if self.run.Ns == 1:
            self.run.dist = dist[..., self.l1, :, :, :]
        else:
            if not hasattr(self.run, 'dist'):
                self.run.dist = [[] for _ in range(self.run.Ns)]
            self.run.dist[s] = dist[..., self.l1, :, :, :]
