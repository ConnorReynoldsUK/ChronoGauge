#!/bin/bash -e
#SBATCH --constraint=intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0-6:00
#SBATCH -c 1
#SBATCH --mem 132Gb
#SBATCH -p ei-medium

source python_miniconda-4.5.12_py3.7_kh
source cuda-7.5.18

/hpc-home/creynold/ati_p/python/software/minicon_con/bin/python3.9 sfs_main_combat.py $1
