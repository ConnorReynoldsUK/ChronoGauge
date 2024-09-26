import pandas as pd
import numpy as np
import sys
from functools import  reduce
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import random
from datetime import datetime
import pickle
import argparse


def cyclic_time(times):
    #this is used to convert the target (time of sampling) in hours to cosine and sine values
    times = times % 24
    t_cos = -np.cos((2 * np.pi * times.astype('float64') / 24)+(np.pi/2))
    t_sin = np.sin((2 * np.pi * times.astype('float64') / 24)+(np.pi/2))

    t_circular = np.concatenate((np.asarray(t_cos).reshape(-1, 1), np.asarray(t_sin).reshape(-1, 1)), axis=1)


    return t_circular


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
        i_keep = i_phase.index[:n_genes_bin-1]
        balanced_list.append(i_keep)

    # prior information of the balanced gene list
    balanced_features = prior_info.loc[np.concatenate(balanced_list)]
    balanced_features['phase_bin'] = [float(i) for i in balanced_features['phase_bin']]

    # select a gene at random from the balanced features
    first_gene = balanced_features.index[random.randint(0, balanced_features.shape[0]-1)]
    print('Feature selection will initiate with gene: {}'.format(first_gene))


    return balanced_features, first_gene

