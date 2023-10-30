#!/bin/bash
#PBS -q LARGE
#PBS -N prepro_tekdd
#PBS -l select=1
#PBS -j oe


source /etc/profile.d/modules.sh
module load singularity/3.9.5
cd ${/home/s2316002/Project2}
singularity exec /home/s2316002/Project2/runcode.sif python /home/s2316002/capstone_project/kdd/dataset/preprocess_df_test.py
