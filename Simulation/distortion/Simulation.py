"""
Simulates a particle rocking through an N-shaped and noisy curve.
"""


import numpy as np
import ptypy
import nmutils
from nmutils.utils.bodies import TruncatedOctahedron, Cube
from bcdiass.utils import roll
import matplotlib.pyplot as plt
import matplotlib

Nk = 21

#strain size
strain_size = 0.5
strain_type = 'body'



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

photons_in_central_frame=5e6


#original rocking curve and with fluctuactions (noise) with different level

theta0 = np.linspace(-0.4,.4, Nk)

step_size = theta0[1]-theta0[0]
levels = [19.5]


for i in range(len(levels)):


    distortion = (np.random.rand(Nk) - .5)
    distortion = np.fft.ifft(np.fft.fft(distortion)* (np.abs(np.arange(Nk))<Nk//4))
    distortion = np.real(distortion)
    distortion = distortion / np.abs(distortion).max() * step_size * levels[i] * 3.1 #3.x here is the coefficients
    ave_dis = np.sum(np.abs(distortion))/len(distortion)
    
    noise_level = np.floor(ave_dis/step_size * 100).astype(np.int32)
    
    #period, ampl = Nk//2, .02
    #envelope = (1 - np.abs(np.arange(Nk)/(Nk//2) - 1))
    pos_n = len(np.where(distortion>0)[0])
    neg_n = len(np.where(distortion<0)[0])
    
    #make the distortion more significane, here is the sin shape

    theta = np.copy(theta0)
    #theta += envelope * np.sin(np.arange(Nk) * 2 * 3.14 / period) * ampl
    theta += distortion

    ampl = 5

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

    ### optionally add strain
    if strain_type == 'body':
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
    elif strain_type == 'surface':
        o.scale(.8)
        v.data[np.where(1 - o.contains((xx[0], -yy[0], zz[0])))] *= np.exp(1j * strain_size )

    # reference rocking curve
    # data_ref = []
    # for ioffset, offset in enumerate(theta0):
    #     g.bragg_offset = offset
    
    #     I = np.abs(g.propagator.fw(v.data)) ** 2
    #     #I = roll(I, phi[ioffset]*0, roll_center)
    #     exit = g.overlap2exit(v.data)
    #     data_ref.append({'offset': offset,
    #                  'angle': angle,
    #                  'diff': I,
    #                  'exit': exit})
    # frames_ref = [d['diff'] for d in data_ref]
    # if photons_in_central_frame is not None:
    #     central = np.argmin(np.abs(theta0))
    #     photons_per_intensity = photons_in_central_frame / frames_ref[central].sum()
    #     global_max = frames_ref[central].max() * photons_per_intensity
    #     noisy_frames = []
    #     for i, frame in enumerate(frames_ref):
    #         mask = frame < 0
    #         frame[mask] = -1
    #         noisy = frame * photons_per_intensity
    #         #noisy = nmutils.utils.noisyImage(frame, photonsTotal=photons_per_intensity * frame.sum())
    #         noisy[mask] = -1
    #         noisy_frames.append(noisy)
    #     frames_ref = noisy_frames
    # else:
    #     central = np.argmin(np.abs(theta0))
    #     global_max = frames_ref[central].max()
    # np.savez_compressed('./sim_ref%i.npz'%noise_level, offsets=theta0, frames=frames_ref, particle=v.data)#roll=phi)



    #rocking curve with noise
    data = []
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
        for i, frame in enumerate(frames):
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
    np.savez_compressed('./sim_%i.npz'%noise_level, offsets=theta, frames=frames, particle=v.data)# rolls=phi * 0)

    data_sum = np.log10(np.sum(frames, axis=(1,2)))
    #data_ref_sum = np.log10(np.sum(frames_ref, axis=(1,2)))
    
    fig, ax0 = plt.subplots(1, 1)

    ax0.set_title('Rocking Curves & Distortion, Pos_n: %i'%pos_n)
    ax0.plot(theta0, data_sum, alpha=0.5)
    #ax0.plot(theta0, data_ref_sum, color='r', alpha=0.8)
    ax0.legend(['Simu.', 'Ref.'])
    ax0.set_xlabel('Principle rocking curve angles (degree)')
    ax0.set_ylabel('Log-integrate intensity of frames')
    
    plt.savefig('./RockingCurve_%i.png'%noise_level)




