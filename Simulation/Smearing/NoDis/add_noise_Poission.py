"""
Transfer the non-noisy diffraction frames into diffractin with Poisson noise
"""


from doctest import Example
import numpy as np
import nmutils
import matplotlib.pyplot as plt
import matplotlib



file_ref = './Example/sim_ref.npz'

filename = './Example/sim_611.npz'
sim_name = filename[14:-4]

frames = np.load(filename)['frames']
particle = np.load(filename)['particle']
offsets = np.load(filename)['offsets']
#phi = np.load(filename)['rolls']

frames_ref = np.load(file_ref)['frames']
theta0 = np.load(file_ref)['offsets']


photons_in_central_frame = 5e6
#central = np.argmin(np.abs(theta0))
central = np.argmin(np.abs(offsets))
photons_per_intensity = photons_in_central_frame / frames_ref[central].sum()
global_max = frames[central].max() * photons_per_intensity

print('now generting the Poisson noise frames')
noisy_frames = []
for i, frame in enumerate(frames):
    mask = frame<0
    frame[mask] = -1
    noise_ = nmutils.utils.noisyImage(frame, photonsTotal=frame.sum() * photons_per_intensity)
    noise_[mask] = -1
    noisy_frames.append(noise_)
    
print('now saving the document')
np.savez_compressed('./data_noise/sim_%s_noise.npz'%sim_name, offsets=offsets, frames=noisy_frames, particle=particle) #, phi = phi)

