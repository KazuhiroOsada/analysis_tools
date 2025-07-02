import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from chunk_reader import coord_reader, bg_reader, field_reader, current_reader, moment_reader, dist_reader


# number of threads for parallel processing if max_workers is None, it will be set automatically
max_workers = None


class DataReader:
    """
    Reader for domain decomposed data files of GEMSIS-RC
    """
    def __init__(self, run):
        self.run = run
        self.domain = run.domain
        self.Nd3, self.Nd2, self.Nd1 = run.domain # number of domain decomposition
        self.N3, self.N2, self.N1, self.Nm, self.Nv = run.N3, run.N2, run.N1, run.Nm, run.Nv
        self.N3_local, self.N2_local, self.N1_local = run.N3//self.domain[0], run.N2//self.domain[1], run.N1//self.domain[2]
        self.ext = '.dat'
        
    def get_file_path(self, name, d1, d2, d3, s=None):
        file_path = os.path.join(self.run.prefix, name + '-{:02d}-{:02d}-{:02d}'.format(d1, d2, d3) + self.ext)
        if s is not None:
            file_path = os.path.join(self.run.prefix, name + '{:d}-{:02d}-{:02d}-{:02d}'.format(s+1, d1, d2, d3) + self.ext)
        return file_path
    
    def thread_parallel_processing(self, process_chunk):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_chunk, d1, d2, d3)
                for d3 in range(self.Nd3)
                for d2 in range(self.Nd2)
                for d1 in range(self.Nd1)
            ]
            for future in futures:
                future.result()
        
    def read_coord(self):
        """
        read 'coord-**-**-**.dat' files for coordinate data and parameters
        """
        shape_global = (self.N3, self.N2, self.N1, 3)
        # scalar and logical grid points
        scalar = np.zeros(4) # a, dx1, dx2, dx3
        x1, x2, x3 = np.zeros(self.N1), np.zeros(self.N2), np.zeros(self.N3)
        # metric and cartesian coordinate at cell center
        metric = np.zeros(shape_global)
        xyzi = np.zeros(shape_global)
        # cartesian coordinate at cell boundary (with different local shape !)
        xyzh = np.zeros((self.N3+1, self.N2+1, self.N1+1, 3))

        def process_chunk(d1, d2, d3):
            file_path = self.get_file_path('coord', d1, d2, d3)
            # local indices for the current domain on the integer grid
            i1_local = np.arange(self.N1_local) + self.N1_local*d1
            i2_local = np.arange(self.N2_local) + self.N2_local*d2
            i3_local = np.arange(self.N3_local) + self.N3_local*d3
            idx = (i3_local[:, None, None], i2_local[None, :, None], i1_local[None, None, :], slice(None))
            # on the half integer grid
            h1_local = np.arange(self.N1_local+1) + self.N1_local*d1
            h2_local = np.arange(self.N2_local+1) + self.N2_local*d2
            h3_local = np.arange(self.N3_local+1) + self.N3_local*d3
            hidx = (h3_local[:, None, None], h2_local[None, :, None], h1_local[None, None, :], slice(None))
            # read data from file
            scalar[:], x1[i1_local], x2[i2_local], x3[i3_local], \
            metric[idx], xyzi[idx], xyzh[hidx] = coord_reader(file_path, self.N1_local, self.N2_local, self.N3_local)
        
        self.thread_parallel_processing(process_chunk)

        # pack data into run object
        self.run.a, self.run.dx1, self.run.dx2, self.run.dx3 = scalar
        self.run.b = np.arcsinh(self.run.a)
        self.run.x1, self.run.x2, self.run.x3 = x1, x2, x3
        self.run.h1, self.run.h2, self.run.h3 = metric[..., 0], metric[..., 1], metric[..., 2]
        # unit is Re for cartesian coordinate
        self.run.Xi, self.run.Yi, self.run.Zi = xyzi[..., 0]/self.run.Re, xyzi[..., 1]/self.run.Re, xyzi[..., 2]/self.run.Re
        self.run.Xh, self.run.Yh, self.run.Zh = xyzh[..., 0]/self.run.Re, xyzh[..., 1]/self.run.Re, xyzh[..., 2]/self.run.Re
        # half integer grid
        self.run.xh1 = self.run.x1 + 0.5*self.run.dx1
        self.run.xh2 = self.run.x2 + 0.5*self.run.dx2
        self.run.xh3 = self.run.x3 + 0.5*self.run.dx3

    def read_bg(self):
        """
        read 'bg-**-**-**.dat' files for background magnetic field and density
        """
        shape_global_B0 = (self.N3, self.N2, self.N1, 3)
        shape_global_Rho0 = (self.N3, self.N2, self.N1)
        # background magnetic field and density
        B0 = np.zeros(shape_global_B0)
        Rho0 = np.zeros(shape_global_Rho0)

        def process_chunk(d1, d2, d3):
            file_path = self.get_file_path('bg', d1, d2, d3)    
            i1_local = np.arange(self.N1_local) + self.N1_local*d1
            i2_local = np.arange(self.N2_local) + self.N2_local*d2
            i3_local = np.arange(self.N3_local) + self.N3_local*d3
            idx = (i3_local[:, None, None], i2_local[None, :, None], i1_local[None, None, :], slice(None))
            B0[idx], Rho0[idx[:-1]] = bg_reader(file_path, self.N1_local, self.N2_local, self.N3_local)

        self.thread_parallel_processing(process_chunk)

        # pack data into run object
        self.run.B0   = B0 * self.run.unitB
        self.run.Rho0 = Rho0 * self.run.unitN

    def read_field(self, trange):
        """
        read 'field-**-**-**.dat' files for magnetic field and electric drift
        """
        tstep = range(*trange)
        Nt = len(tstep)
        shape_global = (self.N3, self.N2, self.N1, 3, Nt)
        # magnetic field and electric drift
        V = np.zeros(shape_global)
        B = np.zeros(shape_global)
        
        def process_chunk(d1, d2, d3):
            file_path = self.get_file_path('field', d1, d2, d3)
            i1_local = np.arange(self.N1_local) + self.N1_local*d1
            i2_local = np.arange(self.N2_local) + self.N2_local*d2
            i3_local = np.arange(self.N3_local) + self.N3_local*d3
            idx = (i3_local[:, None, None], i2_local[None, :, None], i1_local[None, None, :], slice(None), slice(None))
            V[idx], B[idx] = field_reader(file_path, self.N1_local, self.N2_local, self.N3_local, trange)

        self.thread_parallel_processing(process_chunk)

        # pack data into run object
        self.run.B = B * self.run.unitB
        self.run.V = V * self.run.unitV

    def read_current(self, trange):
        """
        read 'current-**-**-**.dat' files for current density
        """
        tstep = range(*trange)
        Nt = len(tstep)
        shape_global = (self.N3, self.N2, self.N1, 3, Nt)
        # 4 types of current density
        Jd = np.zeros(shape_global)
        Jm = np.zeros(shape_global)
        Je = np.zeros(shape_global)
        Jp = np.zeros(shape_global)

        def process_chunk(d1, d2, d3):
            file_path = self.get_file_path('current', d1, d2, d3)
            i1_local = np.arange(self.N1_local) + self.N1_local*d1
            i2_local = np.arange(self.N2_local) + self.N2_local*d2
            i3_local = np.arange(self.N3_local) + self.N3_local*d3  
            idx = (i3_local[:, None, None], i2_local[None, :, None], i1_local[None, None, :], slice(None), slice(None))
            Jd[idx], Jm[idx], Je[idx], Jp[idx] = current_reader(file_path, self.N1_local, self.N2_local, self.N3_local, trange)          

        self.thread_parallel_processing(process_chunk)

        # pack data into run object
        self.run.Jd = Jd * self.run.unitJ
        self.run.Jm = Jm * self.run.unitJ
        self.run.Je = Je * self.run.unitJ
        self.run.Jp = Jp * self.run.unitJ
        self.run.Jtot = (Jd + Jm + Je + Jp) * self.run.unitJ

    def read_moment(self, trange, s=0):
        """
        read 'moment{s+1}-**-**-**.dat' files for moments (density, parallel velocity, parallel pressure, perpendicular pressure)
        """
        tstep = range(*trange)
        Nt = len(tstep)
        shape_global = (self.N3, self.N2, self.N1, Nt) # for each moment values
        # moments
        Rho = np.zeros(shape_global)
        Vpa = np.zeros(shape_global)
        Ppa = np.zeros(shape_global)
        Ppe = np.zeros(shape_global)

        def process_chunk(d1, d2, d3):
            file_path = self.get_file_path('moment', d1, d2, d3, s)
            i1_local = np.arange(self.N1_local) + self.N1_local*d1
            i2_local = np.arange(self.N2_local) + self.N2_local*d2
            i3_local = np.arange(self.N3_local) + self.N3_local*d3
            idx = (i3_local[:, None, None], i2_local[None, :, None], i1_local[None, None, :], slice(None))
            Rho[idx], Vpa[idx], Ppa[idx], Ppe[idx] = moment_reader(file_path, self.N1_local, self.N2_local, self.N3_local, trange)

        self.thread_parallel_processing(process_chunk)

        # pack data into run object
        if self.run.Ns == 1:
            self.run.Rho = Rho * self.run.unitN
            self.run.Vpa = Vpa * self.run.unitV
            self.run.Ppa = Ppa * self.run.unitP
            self.run.Ppe = Ppe * self.run.unitP
        else:
            if not hasattr(self.run, 'Rho'):
                self.run.Rho = [ [] for _ in range(self.run.Ns) ]
                self.run.Vpa = [ [] for _ in range(self.run.Ns) ]
                self.run.Ppa = [ [] for _ in range(self.run.Ns) ]
                self.run.Ppe = [ [] for _ in range(self.run.Ns) ]
            self.run.Rho[s] = Rho * self.run.unitN
            self.run.Vpa[s] = Vpa * self.run.unitV
            self.run.Ppa[s] = Ppa * self.run.unitP
            self.run.Ppe[s] = Ppe * self.run.unitP

    def read_dist(self, trange, s=0):
        """
        read 'dist{s+1}-**-**-**.dat' files for phase space density
        """
        tstep = range(*trange)
        Nt = len(tstep)
        shape_global = (self.N3, self.N2, self.N1, self.Nm, self.Nv, Nt)
        shape_local = (self.N3_local, self.N2_local, self.N1_local, self.Nm, self.Nv)
        elements_local = self.N3_local*self.N2_local*self.N1_local*self.Nm*self.Nv
        # phase space density
        dist = np.zeros(shape_global)

        def process_chunk(d1, d2, d3):
            file_path = self.get_file_path('dist', d1, d2, d3, s)
            i1_local = np.arange(self.N1_local) + self.N1_local*d1
            i2_local = np.arange(self.N2_local) + self.N2_local*d2
            i3_local = np.arange(self.N3_local) + self.N3_local*d3
            idx = (i3_local[:, None, None], i2_local[None, :, None], i1_local[None, None, :], slice(None), slice(None), slice(None))
            dist[idx] = dist_reader(file_path, self.N1_local, self.N2_local, self.N3_local, self.Nm, self.Nv, trange)

        self.thread_parallel_processing(process_chunk)

        # pack data into run object
        if self.run.Ns == 1:
            self.run.dist = dist
        else:
            if not hasattr(self.run, 'dist'):
                self.run.dist = [ [] for _ in range(self.run.Ns) ]
            self.run.dist[s] = dist
