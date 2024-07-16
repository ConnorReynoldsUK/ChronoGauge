import pandas as pd
import numpy as np
import sys
from functools import  reduce
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import random
from sfs_class_notebook import SFS_hub
import time
import pickle

# change to any number 0-9 if not using command line
# gene_number = int(sys.argv[0])
gene_number = int(sys.argv[1])

def main_gene_selector(main_gene, df, df_copy):
    if main_gene not in df.index:
        df.loc[main_gene] = df_copy.loc[main_gene]
    return df


def cyclic_time(times):
    # this converts any time to a -cosine and sine value
    times = np.asarray(times)
    times = times % 24
    t_cos = -np.cos((2 * np.pi * times.astype('float64') / 24)+(np.pi/2))
    t_sin = np.sin((2 * np.pi * times.astype('float64') / 24)+(np.pi/2))

    return t_cos, t_sin

def phase_ranker(phases, ranks):
    # ranks genes by ryhthmicity q value
    common_genes = [i for i in ranked_genes.index if i in phases.index]
    ranks = ranks.loc[common_genes].sort_values('mean_q').iloc[:15000, 0]
    phases = phases.loc[ranks.index]

    print(ranks)

    return pd.concat((ranks, phases), axis=1)


tf.random.set_seed(0)

N_GENES = 40
SEED = gene_number
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# phases = metacycle defined phase bins
# ranked genes = metacycle q values
phases = pd.read_csv('data/old_bins.csv', index_col=0)
ranked_genes = pd.read_csv('data/old_q.csv', index_col=0)
ranked_genes = ranked_genes.sort_values('mean_q')
drop_out = random.sample(range(0, ranked_genes.shape[0]-1), int(ranked_genes.shape[0]*0.5))

ranked_genes = ranked_genes.iloc[drop_out].sort_values('mean_q')



X_train = pd.read_csv('data/X_combo_original.csv', index_col=0)
Y_train = [float(i.split('.')[0]) for i in X_train.columns]
Y_train_cos, Y_train_sin = cyclic_time(Y_train)
Y_train_data = np.concatenate((np.asarray(Y_train_cos).reshape(-1, 1), np.asarray(Y_train_sin).reshape(-1, 1)), axis=1)










combinds = reduce(np.intersect1d, (ranked_genes.index, X_train.index))


ranked_genes = ranked_genes.loc[combinds]
ranked = phase_ranker(phases, ranked_genes)
X_train = X_train.loc[ranked.index]

X_train_copy = X_train.copy()

# core geenes are used as a seed for building the proxy set

u_phases = np.unique(ranked['phase'])
# number of genes per phase bin used
N_PER_CLUSTER = 25

counts = ranked['phase'].value_counts()
print(counts)
counts = counts.loc[counts > N_PER_CLUSTER].index.values

# we want to make sure the core gene is NOT removed regardless of its rhythmicity rank

keep = []

for i in range(u_phases.shape[0]):
    i_phase = u_phases[i]
    i_ranked = ranked.loc[ranked['phase'] == i_phase]
    keep.append(i_ranked.index[:N_PER_CLUSTER])

keep = np.concatenate(keep)

X_train = X_train.loc[keep]





# normalize all datasets
scaler = StandardScaler()
X_train = pd.DataFrame(data=scaler.fit_transform(X_train.T), index=X_train.columns, columns=X_train.index)
ranked = ranked.loc[X_train.columns]
main_gene = ranked.index[random.randint(0, 50)]

print(main_gene)
sys.exit()
sfs_i = SFS_hub(main_gene, X_train, Y_train_data, X_train, Y_train_data, ranked)
sfs_i.manual_control()

gene_length = 0

array_length = []
array_error = []
if not os.path.exists('results'):
    os.mkdir('results')
if not os.path.exists('results/show_seq'):
    os.mkdir('results/show_seq')
exp_name = time.time()
os.mkdir('results/{}'.format(exp_name))


for n in range(0, 100000000):
    if gene_length < N_GENES:
        gene_iteration, error_iteration = sfs_i.manual_run()
        gene_length = len(gene_iteration)



        array_error.append(error_iteration)
        array_length.append(len(gene_iteration))


        if gene_length > 7:
            gene_iteration, error_iteration = sfs_i.manual_reverse()

        if gene_length > 4:
            if isinstance(error_iteration, list):
                error_iteration = error_iteration[0]

            if error_iteration < 60:
                print('Iteration number: {}'.format(n))
                print('Gene length: {}'.format(gene_length))
                print('Error: {}'.format(error_iteration))

                with open('results/{}/{}_{}_results.p'.format(exp_name, gene_length, error_iteration),
                          'wb') as handle:
                    pickle.dump(gene_iteration, handle)
                with open('results/show_seq/{}.p'.format(exp_name),
                          'wb') as handle:
                    pickle.dump([array_length, array_error], handle)
    else:
        break

# SequentialFeatureSelectionCoreGene(X_train, Y_train_data, ranked, N_GENES, 120, X_yav, Y_yav_data, main_gene, main_gene_idx)
