import utils_gen2 as utils

import numpy as np
from scipy.ndimage import affine_transform, rotate
from skimage.transform import resize
import matplotlib.pyplot as plt

results_path = './quasi2D/Presults'
generator = utils.RandomLattice(path='./generated', supercell=[25,25,1], max_rotation_angle=0)  # [15,15,1]

labels = []
i = 0
lattice, label = generator.generate_lattice(16)
print(label)
crystal_render = generator.render_lattice(lattice)

np.save(results_path+'/lattice', crystal_render)

crystal_render_shift = np.ones_like(crystal_render)*255
ix,iy = 10,20
crystal_render_shift[:-ix,:-iy] = crystal_render[ix:,iy:]

crystal_render_rot_60 = rotate(crystal_render, 60., reshape=False, cval=255)
crystal_render_rot_120 = rotate(crystal_render, 120., reshape=False, cval=255)
crystal_render_rot_180 = rotate(crystal_render, 180., reshape=False, cval=255)

transform_r0, transform_i0 = generator.fourier_trasform_(255-crystal_render[500:1500,500:1500])
transform = transform_r0+transform_i0*1.j

transform_r, transform_i = generator.fourier_trasform_(255-crystal_render_shift[500:1500,500:1500])
transform_shift = transform_r+transform_i*1.j

transform_r, transform_i = generator.fourier_trasform_(255-crystal_render_rot_60[500:1500,500:1500])
transform_rot_60 = transform_r+transform_i*1.j

transform_r, transform_i = generator.fourier_trasform_(255-crystal_render_rot_120[500:1500,500:1500])
transform_rot_120 = transform_r+transform_i*1.j

transform_r, transform_i = generator.fourier_trasform_(255-crystal_render_rot_180[500:1500,500:1500])
transform_rot_180 = transform_r+transform_i*1.j

Phi_shift = np.angle(transform_shift/transform)
Phi_rot_60 = np.angle(transform_rot_60/transform)
Phi_rot_120 = np.angle(transform_rot_120/transform)
Phi_rot_180 = np.angle(transform_rot_180/transform)

plot = utils.Plotting('./quasi2D/Presults')

plot.plot_lattice(crystal_render)
plot.plot_lattice(crystal_render_shift, filename = 'lattice_shift.png')
plot.plot_lattice(crystal_render_rot_60, filename = 'lattice_rot_60.png')
plot.plot_lattice(crystal_render_rot_120, filename = 'lattice_rot_120.png')

plot.plot_phaseF(Phi_shift, filename = 'phaseF_shift.png', select=(400,600))
plot.plot_phaseF(Phi_rot_60, filename = 'phaseF_rot_60.png', select=(400,600))
plot.plot_phaseF(Phi_rot_120, filename = 'phaseF_rot_120.png', select=(400,600))
plot.plot_phaseF(Phi_rot_180, filename = 'phaseF_rot_180.png', select=(400,600))

transform_r0, transform_i0 = transform_r0[350:650,350:650], transform_i0[350:650,350:650]
transform = transform_r0+transform_i0*1.j
# apply the same symmetries but in k-space
# R_{\pi/3}
sym_matrix = utils.symmetry_operations[4].copy()
sym_matrix[:2,:2] = np.linalg.inv(sym_matrix[:2,:2]).T # in F-space: https://bartmcguyer.com/notes/note-8-AffineTheorem.pdf
center = np.array([150, 150])
offset = center - sym_matrix[:2,:2] @ center
sym_matrix[:,2] = offset
print(sym_matrix)
transform_ar = affine_transform(transform_r0, sym_matrix)
transform_ai = affine_transform(transform_i0, sym_matrix)
transform_a = transform_ar+transform_ai*1.j
Phi_a = np.angle(transform_a/transform)
plot.plot_phaseF(Phi_a, filename = 'phaseF_a_60.png', select=(50,250))

# R_{2\pi/3} 
sym_matrix = utils.symmetry_operations[2].copy()
sym_matrix[:2,:2] = np.linalg.inv(sym_matrix[:2,:2]).T
center = np.array([150, 150])
offset = center - sym_matrix[:2,:2] @ center
sym_matrix[:,2] = offset
print(sym_matrix)
transform_ar = affine_transform(transform_r0, sym_matrix)
transform_ai = affine_transform(transform_i0, sym_matrix)
transform_a = transform_ar+transform_ai*1.j
Phi_a = np.angle(transform_a/transform)
plot.plot_phaseF(Phi_a, filename = 'phaseF_a_120.png', select=(50,250))

# R_{\pi} 
sym_matrix = utils.symmetry_operations[1].copy()
sym_matrix[:2,:2] = np.linalg.inv(sym_matrix[:2,:2]).T
center = np.array([150, 150])
offset = center - sym_matrix[:2,:2] @ center
sym_matrix[:,2] = offset
print(sym_matrix)
transform_ar = affine_transform(transform_r0, sym_matrix)
transform_ai = affine_transform(transform_i0, sym_matrix)
transform_a = transform_ar+transform_ai*1.j
Phi_a = np.angle(transform_a/transform)
plot.plot_phaseF(Phi_a, filename = 'phaseF_a_180.png', select=(50,250))

#torch.nn.functional.grid_sample