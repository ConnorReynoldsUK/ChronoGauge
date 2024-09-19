import pandas as pd
import numpy as np
import sys
from functools import  reduce
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import random
from chronogauge.sfs_method_git import SFS_hub
from datetime import datetime
import pickle
from chronogauge.utils import cyclic_time
import argparse



parser = argparse.ArgumentParser(description='An example script which sequentially selects features for circadian time prediction')
parser.add_argument('--seed', type=int, help='Deafault : 0 . Ensures consistency across different scripts. To generate an ensemble, use different seeds for each run.',
                    default=0)
parser.add_argument('--x_training', type=str, help='Define the training expression matrix .csv file. If None, will use defualt training data.',
                    default='data/expression_matrices/x_training.csv')
parser.add_argument('--target_training', type=str, help='Define model targets - sampling times in hours for training data across 1st column of .csv file. If None, use default training targets.',
                    default='data/targets/target_training.csv')
parser.add_argument('--prior_info', type=str, help='Prior information regarding phase bin and meta2d q-value for each feature. If None, use default info.',
                    default='data/sfs_input/sfs_gene_info.csv')
parser.add_argument('--max_genes', type=int, help='Deafault : 40 . Number of gene features in which the algorithm will stop.',
                    default=40)
parser.add_argument('--n_gene_balance', type=int, help='Deafault : 25 . Number of gene features selected from each phase bin.',
                    default=25)
parser.add_argument('--bootstrap_fraction', type=float, help='Deafault : 0.5 . Fraction of genes to be randomly selected in run.',
                    default=0.5)

parser.add_argument('--n_iterations', type=int, help='Deafault : 10000 . Number of iteratations until algorithm stops.',
                    default=10000)
parser.add_argument('--out_results', type=str, help='Directory to output test results in .csv format. If None, will output to results/test_results directory.',
                    default='results/sfs_results')

args = parser.parse_args()


gene_number = args.seed
SEED = gene_number
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.random.set_seed(0)

if not os.path.exists(args.out_results):
    os.mkdir()
# where all possible gene features are output
if not os.path.exists(args.out_results + '/feature_space'):
    os.mkdir(args.out_results + '/feature_space')
# where each iteration is output
if not os.path.exists(args.out_results + '/show_seq'):
    os.mkdir(args.out_results + '/show_seq')
# get date & time as a directory label
exp_name = args.out_results + 'seed_{}_{}'.format(args.seed, datetime.now().strftime('%H%M%S_%d%m%y'))
if not os.path.exists(exp_name):
    os.mkdir(exp_name)

def initial_check():
    """
    Summarizes the inputs to the script.
    """
    print('Sqeuential feature selection run initialized with following parameters:\n')
    print('Seed value: {}.'.format(args.seed))
    print('Training expression matrix: {}.'.format(args.x_training))
    print('Training target values: {}.'.format(args.target_training))
    print('Prior circadian gene information: {}.'.format(args.prior_info))
    print('Algorithm will cease when N genes = {}.'.format(args.max_genes))
    print('Per phase bin, N genes = {} will be selected.'.format(args.n_gene_balance))
    print('{:.1f}% of genes will be randomly selected from all potentail features.'.format(args.bootstrap_fraction*100))
    print('Algorithm will cease when N iterations = {}.'.format(args.n_iterations))
    print('Output directory: {}.'.format(args.out_results))





def load_expression_data(x_train, y_train):
    """
    Loads expression data and sampling time targets used for either model training or testing.

    Parameters
    ----------
    x_train : str
        The file path for the CSV file containing a gene expression matrix
    y_train : str
        The file path for the CSV file containing the target labels. Required for model training.

    Returns
    ----------
    x_data_features : pd.DataFrame
        The training expression matrix where rows are samples and columns are features.
    y_times : np.ndarray or None
         The target labels as hourly time-points in the 24-hour day.
    y_circular : np.ndarray or None
        The target labels transformered as circular cosine and sine values.
    """
    # load expression matrix
    x_data = pd.read_csv(x_train, index_col=0).T
    # scale expression data
    scaler = StandardScaler()
    x_data = pd.DataFrame(data=scaler.fit_transform(x_data), index=x_data.index, columns=x_data.columns)

    #load target data
    y_data = pd.read_csv(y_train, index_col=0).iloc[:, 0]
    y_data = np.asarray([float(i) % 24 for i in y_data])
    y_times = y_data.copy()
    y_circular = cyclic_time(y_data)

    return x_data, y_times, y_circular

def define_feature_space(prior_info, x_train, bootstrap_fracion, n_genes_bin):
    """
    Loads expression data and sampling time targets used for either model training or testing.

    Parameters
    ----------
    prior_info : pd.DataFrame
        Information regarding each gene's circadian parameters including phase bin and meta2d q-values.
    x_train : pd.DataFrame
        The training expression matrix where rows are samples and columns are features.
    bootstrap_fracion : float
        The proportion of gene features to be selected in each sequential feature selection run. Between 0-1.
    n_genes_bin : int
        The number of gene features selected from each phase bin.

    Returns
    ----------
    balanced_features : pd.DataFrame
        The top features from each phase bin as determined by n_genes_bin value following random bootstrap.
    first_gene : str
        A gene feature randomly selected from the balanced feature set that acts to initialize the feature selection list
    """

    #ensure common feature space between training data and prior info
    common_genes = np.intersect1d(prior_info.index, x_train.columns)
    prior_info = prior_info.loc[common_genes]

    prior_info = prior_info.sample(frac=bootstrap_fracion)

    balanced_list = []
    # select top N features from prior info list
    for i in np.unique(prior_info['phase_bin']):
        i_phase = prior_info[prior_info['phase_bin'] == i]
        i_phase = i_phase.sort_values(by='qval')
        # top gene features recorded
        i_keep = i_phase.index[:n_genes_bin]
        balanced_list.append(i_keep)

    # prior information of the balanced gene list
    balanced_features = prior_info.loc[np.concatenate(balanced_list)]
    balanced_features['phase_bin'] = [float(i) for i in balanced_features['phase_bin']]

    # select a gene at random from the balanced features
    first_gene = balanced_features.index[random.randint(0, balanced_features.shape[0]-1)]
    print('Feature selection will initiate with gene: {}'.format(first_gene))


    return balanced_features, first_gene

def main():
    #list inputs to the script
    initial_check()

    # first read the training expression matrix and target values
    x_train, y_times, y_train = load_expression_data(args.x_training, args.target_training)

    # load the prior information - CSV with each gene feature's phase bin & q value
    prior_info = pd.read_csv('data/sfs_input/sfs_gene_info.csv', index_col=0)

    balanced_features, first_gene = define_feature_space(prior_info, x_train, args.bootstrap_fraction, args.n_gene_balance)

    # initialize the wrapper. NOTE if model returns NaN, reduce learning_rate
    sfs_wrapper = SFS_hub(first_gene, x_train, y_train, balanced_features, learning_rate=0.003)
    sfs_wrapper.manual_control()
    gene_length = 0

    array_length = []
    array_error = []
    for n in range(0, 7):
        if gene_length < args.max_genes:
            gene_iteration, error_iteration = sfs_wrapper.manual_run()
            gene_length = len(gene_iteration)

            array_error.append(error_iteration)
            array_length.append(len(gene_iteration))

    if gene_length > 7:
        gene_iteration, error_iteration = sfs_wrapper.manual_reverse()

    if gene_length > 4:
        if isinstance(error_iteration, list):
            error_iteration = error_iteration[0]

            print('Iteration number: {}'.format(n))
            print('Gene length: {}'.format(gene_length))
            print('Error: {}\n'.format(error_iteration))


if __name__ == "__main__":
    main()

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
