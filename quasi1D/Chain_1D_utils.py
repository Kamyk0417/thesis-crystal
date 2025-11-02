import numpy as np
from scipy.fft import fft, fft2, fftshift
import matplotlib.pyplot as plt

import random as rd
import time

class RandomLattice1D:
    
    def __init__(self, supercell_size=1000, sigma=.5):
        self.supercell_size = supercell_size
        self.grid = np.linspace(0., self.supercell_size, num=self.supercell_size+1, endpoint=True)
        self.sigma = sigma

    def generate_lattice_quasi(self, sublattices, offsets):
        density = np.zeros((self.supercell_size+1, len(sublattices)))
        positions = []
        labels = []
        for s, (Rs, offset) in enumerate(zip(sublattices, offsets)):
            no_nodes = int(self.supercell_size/Rs)
            for i in range(no_nodes+1):
                density[:,s] += np.exp(-((self.grid-i*Rs-offset)/self.sigma)**2/2.)
                positions.append(i*Rs+offset)
                labels.append(s)
        self.density  = density
        sorted_inds = np.argsort(positions)
        self.positions = np.array(positions)[sorted_inds]
        self.labels = np.array(labels)[sorted_inds]
        return self.density
    
    def fourier_transform(self, real_image):
        trans = fftshift(fft(real_image))
        return np.real(trans), np.imag(trans) 


def gen_data(no_samples, sub_max):
    phis = []
    coefs = []
    lat_no = []
    
    start = time.time()
    for j in range(no_samples):
        base_c = rd.uniform(10,30)
        no_subl = rd.randint(1, sub_max)
        #offs = [rd.uniform(0,base_c) for i in range(no_subl+1)]
        offs = [0.0 for i in range(no_subl+1)]
        coefs = [4*rd.random() for i in range(no_subl)]
        coefs.append(1)
        phis.append(phi_gen(base_c, coefs, offsets=offs))
        lat_no.append(len(coefs))
        if j%1000==0:
            stop = time.time()
            print(f'{j} --> {stop-start} s')

    lat_no = [ln-2 for ln in lat_no]
    '''
    labels = []
    for n in lat_no:
        lab = [0 for i in range(sub_max)]
        lab[n] = 1
        labels.append(lab)
    phis = np.array(phis)
    labels = np.array(labels)
    '''
    lat_no = np.array(lat_no)
    return phis, lat_no

def phi_gen(base_const, coefs, offsets):
    lattice = RandomLattice1D()
    density = lattice.generate_lattice_quasi(sublattices=[base_const*c for c in coefs], offsets=offsets)

    ft_r, ft_i = lattice.fourier_transform(density.sum(axis=1))

    Phi = np.angle(ft_r+ft_i*1.j)
    #Phi[abs(ft_r/100)<.01] = 0
    return Phi

'''
lattice = RandomLattice1D(sigma=.5, path='machine_learning\quasicrystals\images')
ft_r, ft_i, ft_p = [], [], []
no_g = 100
i=0
#fig, ax = plt.subplots(2,3, figsize=(12,6))
for g in np.linspace(np.sqrt(2)-.5, np.sqrt(2)+.5, num = int(no_g)+1, endpoint = True):

    density = lattice.generate_lattice_quasi(sublattices=[20,20*g], offsets=[0.,0.])
    density = np.concatenate((density[:500][::-1],density[:501]))
    transform_r, transform_i = lattice.fourier_transform(density.sum(axis=1))

    ft_r.append(transform_r)
    ft_i.append(transform_i)
    
    Phi = np.angle((transform_r+transform_i[::-1]*1.j)/(transform_r+transform_i*1.j))
    Phi[abs(transform_r/100)<.02] = 0
    ft_p.append(Phi)
    ''''''
    if i%(no_g/5) == 0:
        if i<no_g/2:
            ax[0][int(5*i/no_g)].plot(Phi)
            ax[0][int(5*i/no_g)].set_title(f'{g}')
        else:
            ax[1][int(5*(i-3*no_g/5)/no_g)].plot(Phi)
            ax[1][int(5*(i-3*no_g/5)/no_g)].set_title(f'{g}')
    i+=1
    

plt.tight_layout()
plt.show()
''''''
transforms_r = np.array(ft_r)
transforms_i = np.array(ft_i)
transforms_p = np.array(ft_p)
fig, ax = plt.subplots()
#splot = ax.imshow(transforms_p, norm=SymLogNorm(linthresh=1.), cmap='PiYG')
splot = ax.imshow(transforms_p, cmap='PiYG')

ax.plot([0,1000], [(1./2.)*no_g,(1./2.)*no_g], lw=.5)
ax.plot([0,1000], [(1./3.)*no_g,(1./3.)*no_g], lw=.5)
#ax.plot([0,1000], [(np.sqrt(2.)-1.)*no_g,(np.sqrt(2.)-1.)*no_g], lw=.5)
ax.plot([0,1000], [(1./4.)*no_g,(1./4.)*no_g], lw=.5)
ax.plot([0,1000], [(1./5.)*no_g,(1./5.)*no_g], lw=.5)
ax.plot([0,1000], [(1./10.)*no_g,(1./10.)*no_g], lw=.5)

ax.set_aspect(10.)
fig.colorbar(splot)
plt.show()
'''