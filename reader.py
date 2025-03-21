import os

import numpy as np
import matplotlib.pyplot as plt

class DataReader:
    def __init__(self, run, name):
        self.run = run
        self.domain = run.domain
        self.local_shape = (run.N1//self.domain[0], run.N2//self.domain[1], run.N3//self.domain[2])
        self.ext = '.dat'
        if name == 'coord':
            self.read_coord()
        elif name == 'bg':
            self.read_bg()
        elif name == 'field':
            self.read_field()
        elif name == 'current':
            self.read_current()
        elif name == 'moment':
            self.read_moment()
        elif name == 'dist':
            self.read_dist()
        else:
            raise ValueError(f"{name} should be one of ['coord', 'bg', 'field', 'current', 'moment', 'dist']")
        
    def get_file_path(self, name, d1, d2, d3, s=None):
        file_path = os.path.join(self.run.prefix, name + '-{:02d}-{:02d}-{:02d}'.format(d1, d2, d3) + self.ext)
        if s is not None:
            file_path = os.path.join(self.run.prefix, name + '{:d}-{:02d}-{:02d}-{:02d}'.format(s, d1, d2, d3) + self.ext)
        return file_path
        
    def read_coord(self):
        pass

    def read_bg(self):
        pass

    def read_field(self):
        pass

    def read_current(self):
        pass

    def read_moment(self):
        pass

    def read_dist(self):
        pass
