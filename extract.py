from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt

from base import Run

def get_parser():
    parser = OptionParser()
    # options -r, -o, -t, -v, -x
    parser.add_option("-r", "--rundir", dest="rundir", default=None,
                      type="string",
                      help="run directory")
    parser.add_option("-o", "--output", dest="output", default=None,
                      type="string",
                      help="output file")
    parser.add_option("-t", "--trange", dest="trange", default=None,
                      type="int", nargs=3,
                      help="time step range (begin, end, interval)")
    parser.add_option("-v", "--vtransf", dest="vtransf", default=False,
                      action="store_true",
                      help="transform vectors to cartesian coordiante")
    parser.add_option("-x", "--extract", dest="extract", default=None,
                      type="choice", choices=['grid', 'psd', 'mayavi'],
                      help="one of grid, psd, mayavi")
    opts, _ = parser.parse_args()
    # check option consistency
    if opts.rundir is None:
        print('-r option is required')
        parser.print_help()
    if opts.output is None:
        print('-o option is required')
        parser.print_help()
    if opts.extract is None:
        print('-x option is required')
        parser.print_help()
    if opts.extract != 'grid' and opts.trange is None:
        print('-t option is required for data extraction except for grid')
        parser.print_help()
    return opts 

def main():
    opts = get_parser()
    print(opts)

if __name__ == "__main__":
    main()