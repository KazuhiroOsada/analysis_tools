import sys
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt

from coordinate import ModifiedDipole


savefile = 'grid_structure.pdf'

# Plot setting for Case 1
Lmin = 3.6
Lmax = 7.6
Rmin = 3.0
theta_min = np.pi/3
N1 = 64
N2 = 42
stretch = 100

coord = ModifiedDipole(Lmin, Lmax, Rmin, theta_min, N1, N2, stretch)
        
fig, ax = plt.subplots()
for i1 in range(coord.N1):
    ax.plot(coord.x[i1], coord.z[i1], 'k-')
for i2 in range(coord.N2):
    ax.plot(coord.x[:,i2], coord.z[:,i2], 'k-')
ax.set_aspect('equal')
ax.set_xlabel('R [Re]', fontsize=18)
ax.set_ylabel('Z [Re]', fontsize=18)
plt.savefig(savefile, bbox_inches='tight')
