import numpy as np
import matplotlib.pyplot as plt


# TODO?: add VectorTransformer class to convert vector in dipole coordinate to spherical coordinate

class ModifiedDipole:
    """
    Modified Dipole Coordinate (Kageyama et al. 2006) is defined from variable transformation from the standard dipole coordinate.
    """
    def __init__(self, Lmin, Lmax, Rmin, theta_min, N1, N2, stretch):
        """
        Parameters
        ----------
        Lmin, Lmax : L value range [Re]
        Rmin       : minimum radial distance [Re]
        theta_min  : minimum colatitude [rad]
        N1, N2     : number of grid points in x1 and x2 direction
        stretch    : stretching parameter for x1 coordinate [Re^2]
        """
        self.Lmin = Lmin
        self.Lmax = Lmax
        self.Rmin = Rmin
        self.theta_min = theta_min
        self.N1 = N1
        self.N2 = N2
        self.stretch = stretch
        self.Re = 6.378e6
        x1 = self.get_x1_axis()
        x2 = self.get_x2_axis()
        self.x1, self.x2 = np.meshgrid(x1, x2, indexing='ij')
        self.r, self.theta = self.transform2polar(self.x1, self.x2)
        # transform to cartesian coordinate
        self.x = self.r*np.sin(self.theta)
        self.z = self.r*np.cos(self.theta)
    
    def get_x1_axis(self):
        """
        x1 = arcsinh(a*-cos(theta)/r^2)/a_bar

        Returns
        -------
        x1 : 1D array of x1 axis
        """
        a = self.stretch * self.Rmin**2
        a_bar = np.arcsinh(a)
        x1_max = np.arcsinh(a*-np.cos(self.theta_min)/(self.Rmin*self.Re)**2)/a_bar
        return np.linspace(-x1_max, x1_max, self.N1)

    def get_x2_axis(self):
        """
        x2 = sqrt(1/L)

        Returns
        -------
        x2 : 1D array of x2 axis
        """
        x2min = np.sqrt(1/self.Re/self.Lmax)
        x2max = np.sqrt(1/self.Re/self.Lmin)
        return np.linspace(x2min, x2max, self.N2)

    def transform2polar(self, x1, x2):
        """
        Transform dipole coordinate to polar coordinate
        """
        a = self.stretch * self.Rmin**2
        a_bar = np.arcsinh(a)
        mu = np.sinh(a_bar*x1)/a
        chi = x2**2
        zeta = (mu/chi**2)**2
        c1 = 2**(7/3)*3**(-1/3)
        c2 = 2**(1/3)*3**(2/3)
        gamma = (9*zeta+np.sqrt(3*(27*zeta**2+256*zeta**3)))**(1/3)
        w = -c1/gamma+gamma/c2/zeta
        u = -0.5*np.sqrt(w)+0.5*np.sqrt(-w+2/zeta/np.sqrt(w))
        r = u/chi
        theta = np.arcsin(np.sqrt(u))
        theta[np.where( x1 < 0 )] = np.pi - theta[np.where( x1 < 0 )]
        return r/self.Re, theta
    
    def draw_RZ_plane(self):
        """
        draw grid on R-Z plane
        """
        fig, ax = plt.subplots()
        for i1 in range(self.N1):
            ax.plot(self.x[i1], self.z[i1], 'k-')
        for i2 in range(self.N2):
            ax.plot(self.x[:,i2], self.z[:,i2], 'k-')
        ax.set_aspect('equal')
        ax.set_xlabel('R [Re]')
        ax.set_ylabel('Z [Re]')
        plt.show()


class VectorTransformer:
    """
    Convert vector in dipole coordinate to cartesian coordinate
    """
    def __init__(self, x, y, z):
        """
        Parameters
        ----------
        x, y, z: coordinate in cartesian coordinate
        """
        r    = np.sqrt(x**2 + y**2 + z**2)
        r2   = np.sqrt(x**2 + y**2)
        cost = z / r
        sint = np.sqrt(1 - cost**2)
        cosp = x / r2
        sinp = y / r2
        gam  = np.sqrt(1 + 3*cost**2)
        # transform matrix
        self.R = np.zeros(x.shape + (3,3), 'd')
        self.R[...,0,0] = 3*sint*cost*cosp / gam
        self.R[...,0,1] = (-1+3*cost**2)*cosp / gam
        self.R[...,0,2] =-sinp
        self.R[...,1,0] = 3*sint*cost*sinp / gam
        self.R[...,1,1] = (-1+3*cost**2)*sinp / gam
        self.R[...,1,2] = cosp
        self.R[...,2,0] = (-1+3*cost**2) / gam
        self.R[...,2,1] =-3*sint*cost / gam
        self.R[...,2,2] = 0

    def __call__(self, vec):
        """
        vec: vector in dipole coordinate
        """
        xyz = np.zeros_like(vec)
        xyz[...,0] = \
            self.R[...,0,0]*vec[...,0] + \
            self.R[...,0,1]*vec[...,1] + \
            self.R[...,0,2]*vec[...,2]
        xyz[...,1] = \
            self.R[...,1,0]*vec[...,0] + \
            self.R[...,1,1]*vec[...,1] + \
            self.R[...,1,2]*vec[...,2]
        xyz[...,2] = \
            self.R[...,2,0]*vec[...,0] + \
            self.R[...,2,1]*vec[...,1] + \
            self.R[...,2,2]*vec[...,2]
        return xyz


if __name__ == "__main__":
    # simulation setting of Amano et al. 2010
    Lmin = 3.6
    Lmax = 6.6
    Rmin = 3.0
    theta_min = np.pi/3
    N1 = 64
    N2 = 32
    stretch = 100

    coord = ModifiedDipole(Lmin, Lmax, Rmin, theta_min, N1, N2, stretch)
    coord.draw_RZ_plane()
