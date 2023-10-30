#!/bin/bash
#PBS -q SMALL
#PBS -N prepro_cicdf
#PBS -l select=1
#PBS -j oe


source /etc/profile.d/modules.sh
module load singularity/3.9.5
cd ${/home/s2316002/Project2}
singularity exec /home/s2316002/Project2/runcode.sif python /home/s2316002/capstone_project/cic/dataset/preprocess_df.py
