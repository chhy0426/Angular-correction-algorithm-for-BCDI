"""
Fourier shell correlation of reconstructed subsets.

Also rectifies and resamples the full reconstruction on a regular grid
with aspec ratio 1:1:1, for easier visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
from bcdiass.utils import rectify_sample
import os
#plt.ion()

# shifting makes a difference so think about aligning
# the phases have to be aligned to get the q=0 normalization right
# pre-envoloping the particle needed perhaps


strg_pre = '281_ass'
def load(fn, N, cutoff=.1):
    with h5py.File(fn, 'r') as fp:
        p = fp['entry_1/data_1/data'][0]
    p[np.abs(p) < np.abs(p).max() * cutoff] = 0
    p = np.pad(p, N//2, mode='constant')
    com = np.sum(np.indices(p.shape) * np.abs(p), axis=(1,2,3)) / np.sum(np.abs(p))
    com = np.round(com).astype(int)
    p = p[com[0]-N//2:com[0]+N//2,
          com[1]-N//2:com[1]+N//2,
          com[2]-N//2:com[2]+N//2,]
    p[:] = p * np.exp(-1j * np.angle(p[N//2, N//2, N//2]))
    # we also have to make a mask to properly be able to count the pixels
    diff = np.load(os.path.join(os.path.dirname(fn), 'prepared_%s.npz'%strg_pre))['data']
    print(diff.shape)
    com = np.sum(np.indices(diff.shape) * diff, axis=(1,2,3)) / np.sum(diff)
    com = np.round(com).astype(int)
    print(com)
    diff = diff[com[0]-N//2:com[0]+N//2,
                com[1]-N//2:com[1]+N//2,
                com[2]-N//2:com[2]+N//2,]
    mask1d = np.sign(np.max(diff, axis=(1,2)))
    mask = np.ones_like(diff)
    mask = mask * mask1d.reshape((N, 1, 1))
    print(mask1d.shape, mask.shape)
    return p, mask
#files = [f for f in os.listdir('./81/uncrop/') if ('modes_' in f ) and (f.endswith('.h5'))]
file = 'modes_281_ass_15.h5'
folder = './281/uncrop'
#for file in files:
strg = file.split('.h5')[0].split('modes_')[1]
#load the q space scales and work out the new q range after cropping etc
N = 100
# Q3, Q1, Q2 = np.load(folder + 'assem_281.npz')['Q']  # full range of original assembly
# # Q3 = Q3 * 10
# nq3, nq1, nq2 = np.load(folder + 'assem_281.npz')['W'].shape  # original shape
# dq3, dq1, dq2 = Q3 / nq3, Q1 / nq1, Q2 / nq2  # original q space pixel sizes
# N_recons = np.load(folder + 'crop/prepared_281_crop.npz')['data'].shape
# Q3, Q1, Q2 = np.array((dq3, dq1, dq2)) * N_recons  # full q range used in the reconstruction
# # resolution: res * qmax = 2 pi
# #   - if qmax is half the q range (origin to edge) then res is the full period resolution
# #   - if qmax is the full q range (edge to edge) then res is the pixel size
# dr3, dr1, dr2 = (2 * np.pi / q for q in (Q3, Q1, Q2))  # half-period res (pixel size)
# dr3 = dr3 * 0.4


dr3, dr1, dr2 = 10e-09/4, 4.57e-09/2, 4.40e-09

# rectify the full reconstruction, save and plot
p, mask = load(folder+'/modes_%s.h5'%strg, N)
#p = np.flip(p, axis=0)
p, psize = rectify_sample(p, (dr3, dr1, dr2), 15.0, find_order=True) # x z y
np.savez('./281/uncrop/rectified_%s.npz'%strg, data=p, psize=psize)

# plot along all three axes to understand the aspect ratio
fig, ax = plt.subplots(ncols=3)
ext = np.array((-psize/2*N, psize/2*N, -psize/2*N, psize/2*N)) * 1e9 # valid after the operations below
# from the front
ax[0].imshow(np.abs(p).sum(axis=0), extent=ext)
plt.setp(ax[0], xlim=[-50,50], ylim=[-50,50], title='front view',
         xlabel='y', ylabel='z')
# from the top
im = np.flip(np.abs(p).sum(axis=1), axis=0)
ax[1].imshow(im, extent=ext)
plt.setp(ax[1], xlim=[-50,50], ylim=[-50,50], title='top view',
         xlabel='y', ylabel='x')
# from the side
im = np.transpose(im)
ax[2].imshow(im, extent=ext)
plt.setp(ax[2], xlim=[-50,50], ylim=[-50,50], title='side view',
         xlabel='x', ylabel='z')
fig.suptitle('resampled on orthogonal grids')
plt.savefig('./resampled_%s.png'%strg_pre, dpi=300)
