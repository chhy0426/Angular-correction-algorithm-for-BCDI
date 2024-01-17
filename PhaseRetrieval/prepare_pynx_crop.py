"""
Does not consider the q-space pixel sizes, just shifts the COM
to the center and pads to make the third dimension as long as
the other two.
"""

import h5py
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
import matplotlib.pyplot as plt



file_ref = './sim_ref.npz'
file = './assem.npz'

data_ref = np.load(file_ref)['frames']
theta0 = np.load(file_ref)['offsets']

data = np.load(file)['W']
data = np.flip(data, axis=0)

#data = np.load(file)['frames']
#data = np.flip(data, axis=0)

data_ref_sum = np.log10(np.sum(data_ref, axis=(1,2)))
data_sum = np.log10(np.sum(data, axis=(1,2)))


fig, ax0= plt.subplots(1, 1)

theta01 = np.linspace(-0.40, 0.40, data.shape[0])

ax0.set_title('Rocking Curves')
ax0.plot(theta0, data_ref_sum, color='slategrey', alpha=1 )
ax0.plot(theta01, data_sum, color='m', alpha=1 )
ax0.legend(['Ref.','Corrected.'], frameon=False)
ax0.set_xlabel('Principle Rocking curve angles (degree)')
ax0.set_ylabel('Log-integrated intensity')

plt.tight_layout()
plt.savefig('./assem_crop.png', dpi=600)

threshold = 0.3
totaln = threshold * 2 / (np.abs(theta01[0]) + theta01[-1]) * data.shape[0]
max_pos = np.argmax(data_sum)
start = np.int(max_pos - totaln//2)
end = np.int(max_pos + totaln//2)

data = np.load(file)['W'][start:end]
# pad to equal side, it gets weird otherwise
pad = data.shape[-1] - data.shape[0]
before = (pad + 1) // 2
after = pad // 2
data = np.pad(data, ((before, after), (0, 0), (0, 0)), mode='constant', constant_values=0)

np.savez('./prepared_crop.npz', data=np.round((data*100)).astype(int))
