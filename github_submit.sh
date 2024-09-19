#!/bin/bash -e
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem 32Gb
#SBATCH -p ei-medium
#SBATCH -J load_chrono
#SBATCH -o chrono.%j.out
#SBATCH -e chrono.%j.err


/hpc-home/creynold/ati_p/python/software/minicon_con/bin/python3.9 train_model_new.py --x_test data/expression_matrices/x_test_rna.csv --target_test data/targets/target_test_rna.csv --out_model results/saved_model
