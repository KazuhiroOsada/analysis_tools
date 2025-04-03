import os
import warnings
from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt

from base import Run
from coordinate import VectorTransformer


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

class Snapshot(Run):
    def __init__(self, trange, prefix='.', outdir='figure'):
        super().__init__(prefix)
        self.set_trange(trange)
        self.read('coord')
        self.read_equatorial('bg')
        self.read_equatorial('field')
        self.read_equatorial('current')
        self.read_equatorial('moment')
        self.outdir = outdir
        transformer = VectorTransformer(self.Xi[:,:,self.N1//2], self.Yi[:,:,self.N1//2], self.Zi[:,:,self.N1//2])
        self.B0 = transformer(self.B0)
        for it in range(self.Nt):
            self.B[..., it] = transformer(self.B[..., it])
            self.V[..., it] = transformer(self.V[..., it])
        
    def draw_equatorial(self, z, fig, ax,
                        vec = None, vmin=None, vmax=None,
                        log = False, title=None, width=None,
                        cmap = 'jet', colorbar=True, clabel=None):
        """
        z: (N3, N2) array
        vec: (N3, N2, 3) array in CARTESIAN coordinate
        log: if True, use log scale for colormap
        vmin, vmax: min and max value for colormap
        clabel: label for colorbar
        """
        # pcolormesh
        if log:
            import matplotlib.colors as mcolors
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            pcm = ax.pcolormesh(self.Xh[:,:,self.N1//2], self.Yh[:,:,self.N1//2], z,
                                norm=norm, cmap=cmap)
        else:
            pcm = ax.pcolormesh(self.Xh[:,:,self.N1//2], self.Yh[:,:,self.N1//2], z,
                                vmin=vmin, vmax=vmax, cmap=cmap)
        # colorbar
        if colorbar:
            cbar = fig.colorbar(pcm, ax=ax)
            if clabel is not None:
                cbar.set_label(clabel)
        # vector field
        if not vec is None:
            step2, step3 = 2, 4 # reduce arrow density
            ax.quiver(self.Xi[::step3,::step2,self.N1//2], self.Yi[::step3,::step2,self.N1//2], vec[::step3,::step2,0], vec[::step3,::step2,1],
                      angles='xy', scale_units='xy', scale=np.sqrt(vec[:,:,0]**2+vec[:,:,1]**2).max()*0.45,
                      color='white', alpha=0.7)
        # draw Earth
        theta = np.linspace(0,2*np.pi,101)
        x = np.cos(theta)
        y = np.sin(theta)
        ax.plot(x,y,color='black')
        ax.fill_between(x[25:76],y[25:76],color='black')
        # settings
        if width is None:
            width = np.ceil(self.Xi[0,0,self.N1//2])
        ax.set_aspect('equal')
        ax.set_xlim(-width,width)
        ax.set_ylim(-width,width)
        ax.set_xlabel('X [Re]', fontsize=18)
        ax.set_ylabel('Y [Re]', fontsize=18)
        ax.set_title(title,fontsize=20)

    def draw_snapshot(self, it):
        """
        draw 4 equatorial colormaps (density, Pperp w/ vE, Jphi, dB w/ dB)
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        width = np.ceil(self.Xi[0,0,self.N1//2])
        fig.suptitle(f't = {int(self.time[it]):d} [s]', fontsize=20)
        # (a) density
        self.draw_equatorial(self.Rho[..., it], fig, axes[0, 0],
                             vmin=1, vmax=800, log=True,
                             title='Density [/cc]', width=width)
        axes[0,0].text(-(width+1),width+1,'a',fontsize=24,fontweight='bold')
        # (b) Pperp w/ vE
        self.draw_equatorial(self.Ppe[..., it], fig, axes[0, 1], vec=self.V[..., it],
                             vmin=1e-4, vmax=10, log=True,
                             title='Pperp [nPa]', width=width)
        axes[0,1].text(-(width+1),width+1,'b',fontsize=24,fontweight='bold')
        # (c) Jphi
        self.draw_equatorial(self.Jtot[..., 2, it], fig, axes[1, 0],
                             vmin=-2, vmax=2, cmap='coolwarm',
                             title='Jphi [nA/m$^2$]', width=width)
        axes[1,0].text(-(width+1),width+1,'c',fontsize=24,fontweight='bold')
        # (d) dB w/ dB
        B = self.B0 + self.B[..., it]
        Babs = np.sqrt(B[..., 0]**2 + B[..., 1]**2 + B[..., 2]**2)
        B0abs = np.sqrt(self.B0[..., 0]**2 + self.B0[..., 1]**2 + self.B0[..., 2]**2)
        dB = Babs - B0abs
        self.draw_equatorial(dB, fig, axes[1, 1], vec=self.B[..., it],
                             vmin=-5, vmax=5, cmap='coolwarm',
                             title='dB [nT]', width=width)
        axes[1,1].text(-(width+1),width+1,'d',fontsize=24,fontweight='bold')
        plt.tight_layout()

        try:
            plt.show()
        except:
            pass
        # save figure
        fig.savefig(os.path.join(self.outdir, f'{int(self.time[it]):05d}.png'))
        plt.close(fig)

    def draw_snapshot_all(self):
        for it in range(self.Nt):
            print(f'drawing snapshot {it+1}/{self.Nt}', end='\r')
            self.draw_snapshot(it)
        print()


if __name__ == "__main__":
    opts = get_parser()
    snap = Snapshot(opts.trange, prefix=opts.rundir, outdir=opts.outdir)
    snap.draw_snapshot_all()
