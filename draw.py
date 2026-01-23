import numpy as np
import matplotlib.pyplot as plt


def draw_equatorial(run, z, fig=None, ax=None,
                    vec=None, vmin=None, vmax=None, alpha=1.0,
                    log=False, title=None, width=None,
                    xlabel=True, ylabel=True, ticks_size=14, ticks_step=2.0,
                    cmap = 'jet', colorbar=True, clabel=None, cfs=None,
                    gridline=False, gridline_alpha=0.3, return_pcm=False):
    """
    Draw contour plot of Z(N3,N2) on the equatorial plane

    Parameters
    ----------
    run        : Run object
    z          : (N3, N2) array
    fig, ax    : matplotlib Figure and Axes objects (if None, create new one)
    vec        : (N3, N2, 2) array for vector field (if None, no vector field)
    vmin, vmax : min and max value for colormap
    log        : if True, use log scale for colormap
    title      : title of the plot
    width      : half width of the plot [Re] (if None, set automatically)
    xlabel     : if True, draw xlabel
    ylabel     : if True, draw ylabel
    cmap       : colormap, 'jet' in default
    colorbar   : if True, draw colorbar
    clabel     : label for colorbar
    cfs        : font size for colorbar label
    gridline   : if True, draw grid lines
    """
    if run.Xh.ndim == 3: # in case coord is read in 3D
        Xh = run.Xh[:,:,run.N1//2]
        Yh = run.Yh[:,:,run.N1//2]
        Xi = run.Xi[:,:,run.N1//2]
        Yi = run.Yi[:,:,run.N1//2]
    elif run.Xh.ndim == 2: # in case coord is read in 2D (equatorial)
        Xh = run.Xh
        Yh = run.Yh
        Xi = run.Xi
        Yi = run.Yi

    was_fig_none = fig is None or ax is None
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # colormap
    if log:
        import matplotlib.colors as mcolors
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        pcm = ax.pcolormesh(Xh, Yh, z, norm=norm, cmap=cmap, alpha=alpha)
    else:
        pcm = ax.pcolormesh(Xh, Yh, z, vmin=vmin, vmax=vmax, cmap=cmap)
    if colorbar:
        cbar = fig.colorbar(pcm, ax=ax)
        if clabel is not None:
            cbar.set_label(clabel, fontsize=cfs)
    # vector field
    if vec is not None:
        skip2, skip3 = 2, 4 # reduce arrow density
        ax.quiver(Xi[::skip3,::skip2], Yi[::skip3,::skip2], vec[::skip3,::skip2,0], vec[::skip3,::skip2,1],
                  angles='xy', scale_units='xy', scale=np.sqrt(vec[:,:,0]**2+vec[:,:,1]**2).max()*0.45,
                  color='white', alpha=0.7)

    if gridline:
        for i2 in range(run.N2+1):
            ax.plot(Xh[:,i2], Yh[:,i2], color='black',
                    linewidth=0.8 if i2 == 0 or i2 == run.N2 else 0.4, alpha=gridline_alpha)
        for i3 in range(run.N3):
            ax.plot(Xh[i3,:], Yh[i3,:], color='black', linewidth=0.4, alpha=gridline_alpha)

    # draw earth
    theta = np.linspace(0,2*np.pi,101)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.plot(x,y,color='black')
    ax.fill_between(x[25:76],y[25:76],color='black')

    # settings
    if width is None:
        width = np.ceil(Xi[0,0])
    ax.set_aspect('equal')
    ax.set_xlim(-width,width)
    ax.set_ylim(-width,width)
    if xlabel:
        ax.set_xlabel('X [Re]', fontsize=18)
    if ylabel:
        ax.set_ylabel('Y [Re]', fontsize=18)
    ax.set_title(title,fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=ticks_size)
    eps = 1e-3
    ticks = np.arange(-width, width + eps, ticks_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    if was_fig_none:
        plt.show()

    if return_pcm:
        return pcm

def draw_meridial(run, z, i3, fig=None, ax=None,
                vmin=None, vmax=None,
                log=False, title=None, rlim=(None, None), zlim=None,
                cmap = 'jet', colorbar=True, clabel=None):
    """
    Draw contour plot of Z(N2,N1) on the meridional plane at i3

    Parameters
    ----------
    run       : Run object
    z         : (N2, N1) array
    i3        : index in x3 direction
    fig, ax.  : matplotlib Figure and Axes objects (if None, create new one)
    vmin, vmax: min and max value for colormap
    log       : if True, use log scale for colormap
    title     : title of the plot
    rlim      : (rmin, rmax) for R axis [Re] (if None, set automatically)
    zlim      : half width for Z axis [Re] (if None, set automatically)
    cmap      : colormap, 'jet' in default
    colorbar  : if True, draw colorbar
    clabel    : label for colorbar
    """
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
    run.set_trange(trange)
    run.read('moment')

    draw_meridial(run, run.Ppa[0, :, :, 0], 0)
