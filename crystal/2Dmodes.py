import utils_gen as utils

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from scipy.linalg import eigh


results_path = './2Dmodes_results/'
lattice = utils.RandomLattice2D(supercell_size=800, flake_size=15., sigma=.05, path='./generated2d')

lattice_density = lattice.generate_lattice_graphene(sublattices=[24]*2, offsets=[[0.,0.],[0.,0.]], angles=[-np.pi/6.,0.])


fig, ax = plt.subplots()
fig.set_size_inches(5., 5.)
ax.imshow(lattice_density[:,:].sum(axis=2), cmap='Greys')
ax.set_axis_off()
fig.savefig(results_path+'lattice.png', bbox_inches='tight', dpi=400)
plt.show()
