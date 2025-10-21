import numpy as np
import matplotlib.pyplot as plt


def draw_equatorial(run, z, fig=None, ax=None,
                    vec=None, vmin=None, vmax=None,
                    log=False, title=None, width=None,
                    cmap = 'jet', colorbar=True, clabel=None,
                    gridline=False):
    """
    z: (N3, N2) array
    vec: (N3, N2, 3) array in CARTESIAN coordinate
    log: if True, use log scale for colormap
    vmin, vmax: min and max value for colormap
    clabel: label for colorbar
    """
    if not run.is_read['coord']:
        run.read('coord')
    was_fig_none = fig is None or ax is None
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # colormap
    if log:
        import matplotlib.colors as mcolors
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        pcm = ax.pcolormesh(run.Xh[:,:,run.N1//2], run.Yh[:,:,run.N1//2], z,
                            norm=norm, cmap=cmap)
    else:
        pcm = ax.pcolormesh(run.Xh[:,:,run.N1//2], run.Yh[:,:,run.N1//2], z,
                            vmin=vmin, vmax=vmax, cmap=cmap)
    if colorbar:
        cbar = fig.colorbar(pcm, ax=ax)
        if clabel is not None:
            cbar.set_label(clabel)
    # vector field
    if vec is not None:
        skip2, skip3 = 2, 4 # reduce arrow density
        ax.quiver(run.Xi[::skip3,::skip2,run.N1//2], run.Yi[::skip3,::skip2,run.N1//2], vec[::skip3,::skip2,0], vec[::skip3,::skip2,1],
                  angles='xy', scale_units='xy', scale=np.sqrt(vec[:,:,0]**2+vec[:,:,1]**2).max()*0.45,
                  color='white', alpha=0.7)

    if gridline:
        for i2 in range(run.N2):
            ax.plot(run.Xh[:,i2,run.N1//2], run.Yh[:,i2,run.N1//2], color='black',
                    linewidth=0.8 if i2 == 0 or i2 == run.N2-1 else 0.4, alpha=0.3)
        for i3 in range(run.N3):
            ax.plot(run.Xh[i3,:,run.N1//2], run.Yh[i3,:,run.N1//2], color='black', linewidth=0.4, alpha=0.3)

    # draw earth
    theta = np.linspace(0,2*np.pi,101)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.plot(x,y,color='black')
    ax.fill_between(x[25:76],y[25:76],color='black')

    # settings
    if width is None:
        width = np.ceil(run.Xi[0,0,run.N1//2])
    ax.set_aspect('equal')
    ax.set_xlim(-width,width)
    ax.set_ylim(-width,width)
    ax.set_xlabel('X [Re]', fontsize=18)
    ax.set_ylabel('Y [Re]', fontsize=18)
    ax.set_title(title,fontsize=20)

    if was_fig_none:
        plt.show()

def draw_meridial(run, z, i3, fig=None, ax=None,
                vmin=None, vmax=None,
                log=False, title=None, rlim=(None, None), zlim=None,
                cmap = 'jet', colorbar=True, clabel=None):
    """
    z: (N2, N1) array
    log: if True, use log scale for colormap
    vmin, vmax: min and max value for colormap
    clabel: label for colorbar        
    """
    if not run.is_read['coord']:
        run.read('coord')
    was_fig_none = fig is None or ax is None
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # colormap
    Rh = np.sqrt(run.Xh[i3,:,:]**2 + run.Yh[i3,:,:]**2)
    if log:
        import matplotlib.colors as mcolors
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        pcm = ax.pcolormesh(Rh, run.Zh[i3,:,:], z,
                            norm=norm, cmap=cmap)
    else:
        pcm = ax.pcolormesh(Rh, run.Zh[i3,:,:], z,
                            vmin=vmin, vmax=vmax, cmap=cmap)
    if colorbar:
        cbar = fig.colorbar(pcm, ax=ax)
        if clabel is not None:
            cbar.set_label(clabel)
    # draw earth
    theta = np.linspace(0,2*np.pi,101)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.plot(x,y,color='black')

    # settings
    if rlim[0] is None or rlim[1] is None:
        rlim = (0, np.ceil(Rh.max()))
    if zlim is None:
        zlim = np.ceil(np.max(np.abs(run.Zh[i3,:,:])))

    ax.set_aspect('equal')
    ax.set_xlim(rlim[0],rlim[1])
    ax.set_ylim(-zlim, zlim)
    ax.set_xlabel('R [Re]', fontsize=18)
    ax.set_ylabel('Z [Re]', fontsize=18)
    ax.set_title(title,fontsize=20)
    ax.grid(linestyle='--', alpha=0.5)

    if was_fig_none:
        plt.show()


if __name__ == "__main__":
    from base import Run

    run = Run('../run/case1/')
    run.read('coord')
    trange = tuple(map(int,input().split()))
    run.set_trange(trange) # t = 3600 s
    run.read('moment')

    draw_meridial(run, run.Ppa[0, :, :, 0], 0)
