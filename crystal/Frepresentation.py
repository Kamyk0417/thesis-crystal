import utils_gen as utils

import numpy as np
import matplotlib.pyplot as plt

results_path = './Fresults/'
generator = utils.RandomLattice(path='./generated')

labels = []
i = 0
lattice, label = generator.generate_lattice(1)
crystal_render = generator.render_lattice(lattice)
transform_r, transform_i = generator.fourier_trasform_(255-crystal_render)

fig, ax = plt.subplots()
fig.set_size_inches(5., 5.)
ax.set_aspect(1.)
ax.imshow(crystal_render, cmap='gray')
ax.set_axis_off()
fig.savefig(results_path+'lattice.png', dpi=100)
plt.close()

fig, ax = plt.subplots()
fig.set_size_inches(5., 5.)
ax.set_aspect(1.)
ax.imshow(transform_r[400:600,400:600], cmap='gray')
ax.set_axis_off()
fig.savefig(results_path+'fourierR.png', dpi=100)
plt.close()

fig, ax = plt.subplots()
fig.set_size_inches(5., 5.)
ax.set_aspect(1.)
ax.imshow(transform_i[400:600,400:600], cmap='gray')
ax.set_axis_off()
fig.savefig(results_path+'fourierI.png', dpi=100)
plt.close()

fig, ax = plt.subplots()
fig.set_size_inches(5., 5.)
ax.set_aspect(1.)
ax.imshow(np.abs(transform_r[400:600,400:600]+transform_i[400:600,400:600]*1.j), cmap='gray')
ax.set_axis_off()
fig.savefig(results_path+'fourierA.png', dpi=100)
plt.close()
