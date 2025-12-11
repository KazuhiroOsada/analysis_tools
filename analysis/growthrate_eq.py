import os
import sys
sys.path.append('..')
from time import time

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from growthrate import *


def compute_equatorial_growth_rate(run, cwt_analysis_file, out_dir):
    """
    Parameters
    ----------
    run               : Run object with equatorial 'dist', 'field', 'bg', 'coord' data read
    cwt_analysis_file : path to cwt analysis file
    out_dir           : output directory
    """
    pass

if __name__ == '__main__':
    run = Run('../../run/case1b128')
    run.set_trange((0, 91, 1), 'v')
    t0 = time()
    run.read_equatorial('bg')
    run.read_equatorial('coord')
    run.read_equatorial('field')
    run.read_equatorial('dist')
    run.calc_magnetic_amplitude()
    t1 = time()
    print(f'Data read time: {t1 - t0:.2f} s')


