import sys
import time
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from draw import draw_equatorial


def moving_average_nan(data, width=10):
    """
    Apply moving average filter to data with Nan values like
    np.array([Nan, ..., Nan, x1, x2, ..., xM, Nan, ..., Nan])
    """
    n = len(data)
    result = np.full(n, np.nan)

    start = np.argmax(~np.isnan(data))
    end = n - 1 - np.argmax(~np.isnan(data[::-1]))

    # in case all Nan
    if start > end: 
        return result

    half_width = width // 2
    for i in range(start, end + 1):
        left = max(0, i - half_width)
        right = min(n, i + half_width + 1)
        result[i] = np.nanmean(data[left:right])

    return result 

def 