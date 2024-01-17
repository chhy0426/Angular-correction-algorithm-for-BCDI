"""
Assembles simulated frames from strained particles.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
plt.ion()

from bcdiass.utils import C, M
from bcdiass.utils import generate_initial, generate_envelope, pre_align_rolls
from bcdiass.utils import ProgressPlot, rectify_sample

if __name__ == '__main__':
	# input and parameters

	Nj, Nl, ml = 5, 10, 1
	Nj_max = 25
	fudge = 1e-5
	increase_Nj_every = 3
	increase_fudge_every = 3
	increase_fudge_by = 2**(1/3)
	fudge_max = 8e-4
	num_iter =  80
	filename = 'picked_REAL_DATA.npz'
	data = np.load(filename)['data']
	#data[np.where(data <=40)] = 0
	Dmax_h = 200
	force = 1
	strg = '35_%i_3D' %Dmax_h




	a = 5.6578e-10  # paritcle parameters
	E = 13088.

	psize = 75e-6  # pixel size
	distance = 1.83  # distance to sample


	d = a / np.sqrt(3)  # (111)                  #paritcle parameters
	hc = 4.136e-15 * 3.000e8
	theta = np.arcsin(hc / (2 * d * E)) / np.pi * 180  # half Bragg angle


	# detector plane: dq = |k| * dtheta(pixel-pixel)
	Q12 = psize * 2 * np.pi / distance / (hc / E) * data.shape[-1]
	Q3 = Q12 / 200 * 10                                        
	dq3 = Q3 / Nj
	#dq3 = 2 * np.pi / 8e-9
	Dmax = Dmax_h *1e-9
	print('Q3, Q12 is %e %e'%(Q3, Q12))

	# do the assembly, plotting on each iteration
	data, rolls = pre_align_rolls(data, roll_center=None)
	envelope1 = generate_envelope(Nj, data.shape[-1], Q=(Q3, Q12, Q12), Dmax=(Dmax, Dmax, Dmax), theta=theta)
	#envelope2 = generate_envelope(Nj, data.shape[-1], support=support)
	W = generate_initial(data, Nj, sigma=1)
	p = ProgressPlot()
	errors = []
	for i in range(num_iter):
		print(i)
		W, Pjlk, timing = M(W, data, Nl=Nl, ml=ml, beta=fudge,
		                    force_continuity=(6 if i<50 else 10), nproc=20,
		                    find_direction=(i>10))
		#W, Pjlk, timing = M(W, data, Nl=Nl, ml=ml, beta=fudge,
		#		            force_continuity=force, nproc=20,
		#		            find_direction=(i > 8))
		[print(k, '%.3f'%v) for k, v in timing.items()]
		W, error = C(W, envelope1)#*envelope2)
		errors.append(error)
		p.update(np.log10(W), Pjlk, errors, vmax=1)

		# expand the resolution now and then
		if i and (Nj<Nj_max) and (i % increase_Nj_every) == 0:

			W = np.pad(W, ((1,1),(0,0),(0,0)))
			Nj = W.shape[0]
			Q3 = dq3 * Nj
			envelope1 = generate_envelope(Nj, data.shape[-1], Q=(Q3, Q12, Q12), Dmax=(Dmax, 1, 1), theta=theta)
			#envelope2 = generate_envelope(Nj, data.shape[-1], support=support)

			print('increased Nj to %u'%Nj)

		if i and (fudge < fudge_max) and (i % increase_fudge_every) == 0:
			fudge *= increase_fudge_by
			print('increased fudge to %e'%fudge)


	np.savez('ass_%s.npz'%strg, W=W, Pjlk=Pjlk, rolls=rolls, Q=(Q3, Q12, Q12) , support=support)
	plt.savefig('ass_%s.png'%strg, dpi = 600)
