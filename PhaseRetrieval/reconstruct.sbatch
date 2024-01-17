#!/bin/bash
#
# job time, change for what your job requires
#SBATCH -t 10:00:00
#
# job name
#SBATCH -J pynx_chen
#SBATCH -N 1
#SBATCH -p v100

# write this script to stdout-file - useful for scripting errors
echo $PWD
echo "*******************************************************"
cat $0
echo "*******************************************************"
source /home/hu5471ch/pynx-env/bin/activate


rm -rf raw
mkdir raw

rm *.cxi
pynx-id01cdi.py data=prepared_081_ass_crop.npz nb_run_keep=20 nb_run=1000 support_threshold=0.2 0.6
pynx-cdi-analysis.py *.cxi modes
mv *.cxi raw/
mv l-modes.h5 modes_081_ass_crop.h5
