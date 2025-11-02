import utils_gen2 as utils

import numpy as np
from scipy.ndimage import affine_transform

generator = utils.RandomLattice(supercell=[25,25,1], max_rotation_angle=0)
plot0 = utils.Plotting('./thesis crystal/quasi2D/dataset_aux/class0')
plot1 = utils.Plotting('./thesis crystal/quasi2D/dataset_aux/class1')

for group in range(1,18):
    lattice, label = generator.generate_lattice(group)
    crystal_render = generator.render_lattice(lattice)

    group_symmetries = utils.wallpaper_symmetries[utils.wallpaper_groups[group]["name"]]

    transform_r0, transform_i0 = generator.fourier_trasform_(255-crystal_render[500:1500,500:1500])
    transform_r0, transform_i0 = transform_r0[350:650,350:650], transform_i0[350:650,350:650]
    transform = transform_r0+transform_i0*1.j

    for sym in range(1,14):
        sym_matrix = utils.plane_symmetry_operations[sym].copy()
        sym_matrix[:2,:2] = np.linalg.inv(sym_matrix[:2,:2]).T
        center = np.array([150, 150])
        offset = center - sym_matrix[:2,:2] @ center
        sym_matrix[:,2] = offset
        transform_ar = affine_transform(transform_r0, sym_matrix)
        transform_ai = affine_transform(transform_i0, sym_matrix)
        transform_a = transform_ar+transform_ai*1.j
        Phi_a = np.angle(transform_a/transform)
        if group_symmetries[sym] == 0:
            plot0.plot_phaseF(Phi_a, filename = f'phi_{group}_{sym}_0', select=(50,250))
        else: 
            plot1.plot_phaseF(Phi_a, filename = f'phi_{group}_{sym}_1', select=(50,250))
