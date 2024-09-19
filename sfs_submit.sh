#!/bin/bash -e
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem 32Gb
#SBATCH -p ei-medium
#SBATCH -J load_chrono
#SBATCH -o chrono.%j.out
#SBATCH -e chrono.%j.err


/hpc-home/creynold/ati_p/python/software/minicon_con/bin/python3.9 sfs_main_git.py
