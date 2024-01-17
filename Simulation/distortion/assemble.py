"""
Assembles simulated frames from strained particles.
"""
#%%
import numpy as np
import os
import matplotlib.pyplot as plt
plt.ion()
import time
#%%
from utilbcid import C, M
from utilbcid import generate_initial, generate_envelope, pre_align_rolls
from utilbcid import ProgressPlot, rectify_sample
import os

if __name__ == '__main__':
    # input and parameters

    simfiles = [f for f in os.listdir('./') if ('sim_' in f) and (f.endswith('.npz'))]
    for filename in simfiles:
        
        tstart = time.time()
        co = [250,240,255,260]

        Nj = 21
        Nj_max = 201
        fudge = 1e-5
        increase_Nj_every = 5
        increase_fudge_every = 5
        increase_fudge_by = 2**(1/2)
        for k in range(len(co)):
            fudge_h = 8
            fudge_max = fudge_h * 1e-4
            num_iter = Nj_max + 20
            fc_h = 6
            Dmax_h = co[k]

            data = np.load('./' + filename)['frames']

            # physics
            a = 4.065e-10
            E = 10000.
            d = a / np.sqrt(3)  # (111)
            G = 2 * np.pi / d
            hc = 4.136e-15 * 3.000e8
            theta = np.arcsin(hc / (2 * d * E)) / np.pi * 180
            psize = 55e-6
            distance = .5

            # detector plane: dq = |k| * dtheta(pixel-pixel)
            Q12 = psize * 2 * np.pi / distance / (hc / E) * data.shape[-1]
            R12 = 2 * np.pi / Q12 * 200  # it is tricky
            # Q3 = Q12 / 256 * 15  # about the same Q range as the first in-plane minimum
            # dq3 = Q3 / Nj
            dq3 = 0.8 / data.shape[0] / 180 * np.pi * G 
            Q3 = dq3 *  (Nj-1)

            # G111 = 2 * np.pi / d
            # dq3 = 1 / np.shape(data)[0] / 180 * np.pi * G111 
            # Q3 = dq3 * Nj

            #R3 = 2 * np.pi / Q3 * Nj
            Dmax = Dmax_h *1e-9


            # do the assembly, plotting on each iteration
            #data, rolls = pre_align_rolls(data, roll_center=None)
            envelope1 = generate_envelope(Nj, data.shape[-1], Q=(Q3, Q12, Q12), Dmax=(Dmax, 1, 1), theta=theta)
            # envelope2 = generate_envelope(Nj, data.shape[-1], support=(1, .25, .25))
            W = generate_initial(data, Nj)
            p = ProgressPlot()
            errors = []

            iter_num = 250
             

            
            for i in range(iter_num):
                print(i)
                if i < iter_num-1:
                    W, Pjk, timing = M(W, data,  beta=fudge,
                                        force_continuity=(fc_h if i < 100 else 10), nproc=20,
                                        find_direction=(i > 10))
                    [print(k, '%.3f' % v) for k, v in timing.items()]
                    W, error = C(W, envelope1)  # *envelope2)
                    errors.append(error)
                else:
                    W, Pjk, timing = M(W, data,  beta=fudge,
                                            force_continuity=(fc_h if i < 100 else 10), nproc=20,
                                            find_direction=(i > 10))
                    [print(k, '%.3f' % v) for k, v in timing.items()]
                    W2 = W
                    W, error = C(W, envelope1)  # *envelope2)
                    errors.append(error)

                p.update(np.log10(W), Pjk, errors, vmax=1)

                # expand the resolution now and then
                if i and (Nj < Nj_max) and (i % increase_Nj_every) == 0:
                    W = np.pad(W, ((2, 2), (0, 0), (0, 0)))
                    Nj = W.shape[0]
                    Q3 = dq3 * (Nj - 1)

                    envelope1 = generate_envelope(Nj, data.shape[-1], Q=(Q3, Q12, Q12), Dmax=(Dmax, 1, 1), theta=theta)
                    # envelope2 = generate_envelope(Nj, data.shape[-1], support=(1, .25, .25))
                    print('increased Nj to %u' % Nj)

                if i and (fudge < fudge_max) and (i % increase_fudge_every) == 0:
                    fudge *= increase_fudge_by
                    print('increased fudge to %e' % fudge)

            np.savez('./assem_%s.npz' %strg, W=W, W2=W2, Pjk=Pjk, Q=(Q3, Q12, Q12))
            plt.savefig('./process_%s.png' %strg)
