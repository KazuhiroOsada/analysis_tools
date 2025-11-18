""" Check README.md for details """

from optparse import OptionParser

import numpy as np

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
                      type="choice", choices=['grid', 'psd', 'mayavi', 'wave'],
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

def write_grid(opts):
    run = Run(opts.rundir)
    run.read('coord')
    with open(opts.output, 'wb') as f:
        # array dimensions
        np.array([run.Ns, run.N3, run.N2, run.N1, run.Nm, run.Nv], np.int32).tofile(f)
        # spatial grid points
        gridpoints = np.stack((run.Xi, run.Yi, run.Zi), axis=-1)
        gridpoints.tofile(f)
        # vpara, mu for each species
        for s in range(run.Ns):
            vp = run.vp[s] * run.unitV
            mu = run.mu[s] * run.Qm[s] * 2 * run.unitV**2/run.unitB
            vp.tofile(f)
            mu.tofile(f)

def write_psd(opts):
    run = Run(opts.rundir)
    run.set_trange(opts.trange, 'v')
    run.read('bg')
    run.read('field')
    run.read('dist')
    run.calc_magnetic_amplitude()
    with open(opts.output, 'wb') as f:
        # array dimensions
        np.array([run.Ns, run.N3, run.N2, run.N1, run.Nm, run.Nv], np.int32).tofile(f)
        # |B| and PSD
        for it in range(run.Nt_v):
            print(f'writing data at t = {run.time_v[it]:.2f} [s]', end='\r')
            np.array([run.time_v[it]], np.float64).tofile(f)
            run.Babs[..., it].tofile(f)
            if run.Ns == 1:
                run.dist[..., it].tofile(f)
            else:
                for s in range(run.Ns):
                    run.dist[s][..., it].tofile(f)
        print()

def write_mayavi(opts):
    run = Run(opts.rundir)
    run.set_trange(opts.trange, 'f')
    run.read('coord')
    run.read('bg')
    run.read('field')
    run.read('current')
    run.read('moment') # assumed that Ns = 1
    if opts.vtransf:
        run.transform('field')
        run.transform('current')
    with open(opts.output, 'wb') as f:
        # array dimensions
        np.array([run.N3+1, run.N2, run.N1], np.int32).tofile(f)
        # spatial grid points
        gridpoints = np.stack((run.Xi, run.Yi, run.Zi), axis=-1)
        gridpoints = np.concatenate((gridpoints, gridpoints[:1,:,:,:]), axis=0) # extends N3 to N3+1
        gridpoints.tofile(f)
        # moments, field, and current
        for it in range(run.Nt):
            print(f'writing data at t = {run.time[it]:.2f} [s]', end='\r')
            np.array([run.time[it]], np.float64).tofile(f)
            moments = np.stack([run.Rho[..., it], run.Vpa[..., it], run.Ppa[..., it], run.Ppe[..., it]], axis=-1)
            moments = np.concatenate((moments, moments[:1]), axis=0)
            moments.tofile(f)
            bfield = run.B0 + run.B[..., it]
            bfield = np.concatenate((bfield, bfield[:1]), axis=0)
            bfield.tofile(f)
            vfield = run.V[..., it]
            vfield = np.concatenate((vfield, vfield[:1]), axis=0)
            vfield.tofile(f)
            jfield = run.Jtot[..., it]
            jfield = np.concatenate((jfield, jfield[:1]) , axis=0)
            jfield.tofile(f)
        print()

def write_wave_analysis(opts):
    run = Run(opts.rundir)
    run.set_trange(opts.trange, 'f')
    run.read_equatorial('coord')
    run.read_equatorial('bg')
    run.read_equatorial('field')
    run.read_equatorial('moment') # assumed that Ns = 1
    if opts.vtransf:
        run.transform('field')
    with open(opts.output, 'wb') as f:
        # array dimensions
        np.array([run.N3+1, run.N2, run.N1], np.int32).tofile(f)
        # spatial grid points
        gridpoints = np.stack((run.Xi, run.Yi, run.Zi), axis=-1)
        gridpoints = np.concatenate((gridpoints, gridpoints[:1,:,:]), axis=0) # extends N3 to N3+1
        gridpoints.tofile(f)
        # moments, field, and current
        for it in range(run.Nt):
            print(f'writing data at t = {run.time[it]:.2f} [s]', end='\r')
            np.array([run.time[it]], np.float64).tofile(f)
            moments = np.stack([run.Rho[..., it], run.Vpa[..., it], run.Ppa[..., it], run.Ppe[..., it]], axis=-1)
            moments = np.concatenate((moments, moments[:1]), axis=0)
            moments.tofile(f)
            bfield = run.B0 + run.B[..., it]
            bfield = np.concatenate((bfield, bfield[:1]), axis=0)
            bfield.tofile(f)
            vfield = run.V[..., it]
            vfield = np.concatenate((vfield, vfield[:1]), axis=0)
            vfield.tofile(f)
        print()    
            

if __name__ == "__main__":
    opts = get_parser()
    if opts.extract == 'grid':
        write_grid(opts)
    elif opts.extract == 'psd':
        write_psd(opts)
    elif opts.extract == 'mayavi':
        write_mayavi(opts)
    elif opts.extract == 'wave':
        write_wave_analysis(opts)
