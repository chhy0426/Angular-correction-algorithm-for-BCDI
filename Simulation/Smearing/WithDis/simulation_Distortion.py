"""
Simulates a particle rocking through an N-shaped and noisy curve.
"""


import numpy as np
import ptypy
import nmutils
from nmutils.utils.bodies import TruncatedOctahedron
from bcdiass.utils import roll
import matplotlib.pyplot as plt
import matplotlib
import os

save_root = '/home/hu5471ch/projects/BCDI_test/Distortion/'
save_paths = ['smear', 'distorted']
try:
    os.mkdir(save_root)
except:
    print('The saved folder is already created!')
N0 = 41
smearing_num = 4
level = 2.30

#Experiment parameter
a = 4.065e-10  #golden particle
E = 10000.
d = a / np.sqrt(3) # (111)
hc = 4.136e-15 * 3.000e8
Btheta = np.arcsin(hc / (2 * d * E)) / np.pi * 180
psize = 55e-6
distance = 0.5
diameter = 250e-9
truncation = .69
angle = 30
roll_center = None
strain_size = 0.5
photons_in_central_frame=5e6

g = ptypy.core.geometry_bragg.Geo_BraggProjection(psize=(psize, psize),
    shape=(41, 256, 256), energy=E*1e-3, distance=distance, theta_bragg=Btheta,
    bragg_offset=0.0, r3_spacing=10e-9)

### make the object container and storage
C = ptypy.core.Container(data_type=np.complex128, data_dims=3)
pos = [0, 0, 0]
pos_ = g._r3r1r2(pos)
v = ptypy.core.View(C, storageID='Sobj', psize=g.resolution, coord=pos_, shape=g.shape)
S = C.storages['Sobj']
C.reformat()

o = TruncatedOctahedron(truncation)
#o = Cube()
o.shift([-.5, -.5, -.5])
o.rotate('z', 45)
o.rotate('y', 109.5/2)
o.scale(diameter)
# now we have an octahedron lying down on the its xy plane.
# the conversion between the right handed octahedron's coordinates
# and Berenguer's is just yB = -y
o.rotate('z', angle)
xx, zz, yy = g.transformed_grid(S, input_space='real', input_system='natural')
v.data[:] = o.contains((xx, -yy, zz))


#v.data[np.where(xx[0] * yy[0] * zz[0] > 0)] *= np.exp(1j * strain_size)
from scipy.special import sph_harm
r = np.sqrt(xx**2 + yy**2 + zz**2)
azimuth = np.arctan(yy / (xx + 1e-30))
polar = np.arccos(zz / (r + 1e-30))
l, m = 3, 2
Y = sph_harm(m, l, azimuth, polar).real
R = np.sin(r / (diameter * truncation / 2) * np.pi)
u = Y * R
u = u / u.ptp() * strain_size
v.data *= np.exp(1j * u[0] )

figure = plt.figure()

Nk = N0 * smearing_num
    
theta_0 = np.linspace(-0.4, 0.4, N0)

theta_data = np.linspace(-0.4, 0.4, Nk)
step_size = theta_data[1]-theta_data[0]

distortion = (np.random.rand(Nk) - .5)
distortion = np.fft.ifft(np.fft.fft(distortion)* (np.abs(np.arange(Nk))<Nk//4))
distortion = np.real(distortion)
distortion = distortion / np.abs(distortion).max() * step_size * level * 3.1 #3.x here is the coefficients
ave_dis = np.sum(np.abs(distortion))/len(distortion)

noise_level = np.floor(ave_dis/step_size * 100).astype(np.int32)
print(noise_level)
theta = theta_data + distortion

for i in range(len(save_paths)):
    data = []
    save_path = save_root + save_paths[i]
    try:
        os.mkdir(save_path)
    except:
        print('The saved folder is already created!')
    
    for ioffset, offset in enumerate(theta):
        g.bragg_offset = offset
        I = np.abs(g.propagator.fw(v.data)) ** 2
        #I = roll(I, phi[ioffset] * 0, roll_center)
        exit = g.overlap2exit(v.data)
        data.append({'offset': offset,
                        'angle': angle,
                        'diff': I,
                        'exit': exit})
    frames = [d['diff'] for d in data]
    if photons_in_central_frame is not None:
        central = np.argmin(np.abs(theta))
        photons_per_intensity = photons_in_central_frame / frames[central].sum()
        global_max = frames[central].max() * photons_per_intensity
        noisy_frames = []
        for k, frame in enumerate(frames):
            mask = frame < 0
            frame[mask] = -1
            noisy = frame * photons_per_intensity
            #noisy = nmutils.utils.noisyImage(frame, photonsTotal=photons_per_intensity*frame.sum())
            noisy[mask] = -1
            noisy_frames.append(noisy)
        frames = noisy_frames
    else:
        central = np.argmin(np.abs(theta))
        global_max = frames[central].max()
    if save_paths[i] == 'smear':
        smear_frames = []
        theta_smear = theta[::smearing_num]
        for j in range(0, len(frames), smearing_num):
            frame_ = np.sum(frames[j:j+smearing_num], axis=0)  # Sum of the group
            frame_ = frame_ / smearing_num    # Average of the group
            smear_frames.append(frame_)
        print(len(smear_frames))
        np.savez_compressed(save_path+'/sim_smear_%i.npz'%noise_level, offsets=theta, frames=smear_frames, particle=v.data)
        
        data_sum = np.log10(np.sum(smear_frames, axis=(1,2)))
        plt.plot(theta_0, data_sum)
    else:
        np.savez_compressed(save_path+'/sim_distorted_%i.npz'%noise_level, offsets=theta, frames=frames, particle=v.data)
        
        data_sum = np.log10(np.sum(frames, axis=(1,2)))
        plt.plot(theta_data, data_sum, 'r')

data_sum = np.log10(np.sum(frames, axis=(1,2)))

plt.legend(save_paths)
plt.xlabel('Principle rocking curve angles (degree)')
plt.ylabel('Log-integrate intensity of frames')
plt.tight_layout()

plt.savefig('./RockingCurve_distorted_%i.png'%noise_level)




