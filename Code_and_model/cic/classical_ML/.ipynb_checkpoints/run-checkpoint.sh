#!/bin/bash
#PBS -q LARGE
#PBS -N CICtrainingML
#PBS -l select=1:ncpus=128
#PBS -j oe


source /etc/profile.d/modules.sh
module load singularity/3.9.5
cd ${/home/s2316002/capstone_project}
singularity exec /home/s2316002/capstone_project/runcode.sif python /home/s2316002/capstone_project/cic/classical_ML/classical_ML.py
