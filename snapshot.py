import os
import warnings
from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from coordinate import VectorTransformer
from draw import draw_equatorial


# suppress warnings from snapshot it = 0
warnings.simplefilter('ignore', category=RuntimeWarning)


def get_parser():
    parser = OptionParser()
    # options -r, -o, -t, -v, -x
    parser.add_option("-r", "--rundir", dest="rundir", default=None,
                      type="string",
                      help="run directory")
    parser.add_option("-o", "--outdir", dest="outdir", default=None,
                      type="string",
                      help="output directory")
    parser.add_option("-t", "--trange", dest="trange", default=None,
                      type="int", nargs=3,
                      help="time step range (begin, end, interval)")
    opts, _ = parser.parse_args()
    # check option consistency
    if opts.rundir is None:
        print('-r option is required')
        parser.print_help()
    if opts.outdir is None:
        opts.outdir = 'figure'
    if opts.trange is None:
        print('-t option is required')
        parser.print_help()
    # create output directory if not exist
    if not os.path.exists(opts.outdir):
        os.makedirs(opts.outdir)
    return opts

def draw_snapshot(run, it, outdir='figure'):
    """
    draw 4 equatorial colormaps (density, Pperp w/ vE, Jphi, dB w/ dB)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    width = np.ceil(run.Xi[0,0,run.N1//2])
    fig.suptitle(f't = {int(run.time[it]):d} [s]', fontsize=20)
    # (a) density
    draw_equatorial(run, run.Rho[..., it], fig, axes[0, 0],
                    vmin=1, vmax=800, log=True,
                    title='Density [/cc]', width=width)
    axes[0,0].text(-(width+1),width+1,'a',fontsize=24,fontweight='bold')
    # (b) Pperp w/ vE
    draw_equatorial(run, run.Ppe[..., it], fig, axes[0, 1], vec=run.V[..., it],
                    vmin=1e-4, vmax=10, log=True,
                    title='Pperp [nPa]', width=width)
    axes[0,1].text(-(width+1),width+1,'b',fontsize=24,fontweight='bold')
    # (c) Jphi
    draw_equatorial(run, run.Jtot[..., 2, it], fig, axes[1, 0],
                    vmin=-2, vmax=2, cmap='coolwarm',
                    title='Jphi [nA/m$^2$]', width=width)
    axes[1,0].text(-(width+1),width+1,'c',fontsize=24,fontweight='bold')
    # (d) dB w/ dB
    B = run.B0 + run.B[..., it]
    Babs = np.sqrt(B[..., 0]**2 + B[..., 1]**2 + B[..., 2]**2)
    B0abs = np.sqrt(run.B0[..., 0]**2 + run.B0[..., 1]**2 + run.B0[..., 2]**2)
    dB = Babs - B0abs
    draw_equatorial(run, dB, fig, axes[1, 1], vec=run.B[..., it],
                    vmin=-5, vmax=5, cmap='coolwarm',
                    title='dB [nT]', width=width)
    axes[1,1].text(-(width+1),width+1,'d',fontsize=24,fontweight='bold')
    plt.tight_layout()

    try:
        plt.show()
    except:
        pass
    # save figure
    fig.savefig(os.path.join(outdir, f'{int(run.time[it]):05d}.png'))
    plt.close(fig)
    
def main():
    opts = get_parser()
    run = Run(opts.rundir)
    run.read('coord')
    run.read_equatorial('bg')
    run.set_trange(opts.trange)
    run.read_equatorial('field')
    run.read_equatorial('current')
    run.read_equatorial('moment')
    transformer = VectorTransformer(run.Xi[:,:,run.N1//2], run.Yi[:,:,run.N1//2], run.Zi[:,:,run.N1//2])
    run.B0 = transformer(run.B0)
    for it in range(run.Nt):
        run.B[..., it] = transformer(run.B[..., it])
        run.V[..., it] = transformer(run.V[..., it])
    for it in range(run.Nt):
        print(f'drawing snapshot {it+1}/{run.Nt}', end='\r')
        draw_snapshot(run, it, outdir=opts.outdir)


if __name__ == "__main__":
    main()
