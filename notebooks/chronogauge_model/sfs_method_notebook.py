import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import math
import tqdm
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import normalize
import random
from numpy.linalg import norm
import time
from collections import Counter
import os

def reset_seeds(reset_graph_with_backend=None):
    if reset_graph_with_backend is not None:
        K = reset_graph_with_backend
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        print("KERAS AND TENSORFLOW GRAPHS RESET")  # optional

    np.random.seed(1)
    random.seed(2)
    tf.compat.v1.set_random_seed(3)




class SFS_hub(object):
    def __init__(self, i_gene, X_data, Y_data, rhythmic_scores=None, error_threshold=60, learning_rate=0.003):

        self.i_gene = i_gene
        self.X_data = X_data
        self.Y_data = Y_data


        self.error_threshold = error_threshold
        self.rhythmic_scores = rhythmic_scores
        self.learning_rate = learning_rate
        # colours in this case = phase bins, but they previously represented clusters
        self.phase_bins = np.unique(self.rhythmic_scores['phase_bin'])

        # iteration counters
        self.current_genes = 0
        self.counter = 0

        # gene lists
        self.genes_perm = []
        self.all_past_genes = []
        i_genes = None

        # define phase counts
        self.counts = {}
        for p in range(self.phase_bins.shape[0]):
            self.counts[self.phase_bins[p]] = 0

        if not os.path.exists('Results'):
            os.mkdir('Results')
        self.exp_name = time.time()
        self.folds = KFold(n_splits=5, shuffle=True, random_state=0)

        self.results_record = {'idx': [], 'genes': [], 'train_error': [], 'train_preds': []}
        self.results_iteration = None
        self.results_remove = None

        # scoring
        self.base_score = 9999.99


        self.early_stop = EarlyStopping(patience=50, restore_best_weights=True, monitor='val_loss', mode='min')


    def custom_loss(self, y_true, y_pred):
        return tf.reduce_mean((tf.math.acos(tf.reduce_sum((y_true * y_pred), axis=-1) / (
                    (tf.norm(y_true, axis=1) * tf.norm(y_pred, axis=1)) + tf.keras.backend.epsilon()))))


    def angler(self, ipreds):


        ang = []
        for k in range(ipreds.shape[0]):
            ang.append(math.atan2(ipreds[k, 0], ipreds[k, 1]) / math.pi * 12)

        for l in range(len(ang)):
            if ang[l] < 0:
                ang[l] = ang[l] + 24
        return ang

    def cyclical_loss(self, true, pred):
        true = self.angler(true)
        pred = self.angler(pred)

        true = np.asarray(true) % 24
        pred = np.asarray(pred)


        errors = []
        err = np.absolute(pred - true)

        for i in err:
            if i > 12:
                i = 24 - i
            errors.append(i * 60)

        return np.mean(errors)

    def larger_model(self):
        # lr = 0.00001
        adam = Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

        # create model

        model = Sequential()
        model.add(Dense(32, kernel_initializer='normal', activation='relu'))
        model.add(Dense(128, kernel_initializer='normal', activation='relu'))
        model.add(Dense(512, kernel_initializer='normal', activation='relu'))
        model.add(Dense(2, kernel_initializer='normal'))
        # Compile model
        model.compile(loss=self.custom_loss, optimizer=adam)
        return model

    def phase_selection(self, count_num, used_genes):
        counts = self.counts.copy()

        if count_num >= 1:


            used_genes = self.rhythmic_scores.loc[self.genes_perm]['phase_bin']
            # make sure genes aren't repeated
            # used_genes = used_genes.loc[[i for i in used_genes.index ]]
            for j in used_genes:
                counts[j] += 1

            min_val = min(counts.values())
            min_counts = [int(k) for k, v in counts.items() if v == min_val]
            random.seed()
            colour = random.choice(min_counts)


            genes = self.rhythmic_scores.loc[self.rhythmic_scores['phase_bin'] == int(colour)]
            idx = genes.index




        if count_num == 0:
            np.random.seed()
            used_genes = self.rhythmic_scores.loc[self.genes_perm[0]]['phase_bin']
            # make sure genes aren't repeated
            # used_genes = used_genes.loc[[i for i in used_genes.index ]]
            for j in used_genes:
                counts[j] += 1

            min_val = min(counts.values())
            min_counts = [int(k) for k, v in counts.items() if v == min_val]
            random.seed()
            colour = random.choice(min_counts)

            genes = self.rhythmic_scores.loc[self.rhythmic_scores['phase_bin'] == int(colour)]
            idx = genes.index

        return idx, colour



    def run_model(self, i_gene, X_data, Y_data, type=None):
        X_d = X_data[i_gene].values

        error = 0  # Initialise error
        all_preds = np.zeros((Y_data.shape[0], 2))  # Create empty array

        for n_fold, (train_idx, valid_idx) in enumerate(self.folds.split(X_data, Y_data)):
            X_train, Y_train = X_d[train_idx], Y_data[train_idx]  # Define training data for this iteration
            X_valid, Y_valid = X_d[valid_idx], Y_data[valid_idx]
            reset_seeds()
            model = self.larger_model()
            # batch size = 2

            model.fit(X_train.astype('float32'), Y_train.astype('float32'),
                      validation_data=(X_valid.astype('float32'), Y_valid.astype('float32')),
                      batch_size=4, epochs=250, callbacks=[self.early_stop],
                      verbose=None)  # Fit the model on the training data

            all_preds[valid_idx] = model.predict(X_valid.astype('float32'))
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            model = None
            del model

            # error += self.cyclical_loss(Y_valid.astype('float32'), preds.astype('float32'))  # Evaluate the predictions
        # forward stage
        if type == 'foward':
            print('+ ',i_gene[-1], self.cyclical_loss(Y_data.astype('float64'), all_preds.astype('float64')), '\n')
            self.results_iteration['train_error'].append(
                    self.cyclical_loss(Y_data.astype('float64'), all_preds.astype('float64')))
        # reverse stage
        if type == 'reverse':
            self.results_remove['train_error'].append(
                self.cyclical_loss(Y_data.astype('float64'), all_preds.astype('float64')))




    def add_genes(self, custom_genes):

        self.genes_perm = custom_genes
        self.all_past_genes.append(custom_genes)
        self.i_gene = custom_genes
        self.i_genes, phase = self.phase_selection(self.counter, self.genes_perm)

        self.results_iteration = {k: [] for k, v in self.results_record.items()}
        self.results_iteration['idx'].append(custom_genes)
        self.results_iteration['genes'].append(custom_genes)

        self.run_model(self.i_gene, self.X_data, self.Y_data, 'foward')
        self.genes_perm = self.results_iteration['idx'][self.results_iteration['train_error'].index(min(self.results_iteration['train_error']))]

        # self.genes_perm = self.results_iteration['idx']
        self.base_score = self.results_iteration['train_error'][self.results_iteration['train_error'].index(min(self.results_iteration['train_error']))]


        self.counter += 1

    def manual_control(self):
        used_genes = None
        # start with seed gene first
        if self.counter == 0:
            self.genes_perm = [[self.i_gene]]
            self.all_past_genes.append([self.i_gene])
        remove = False

        #

        return used_genes

    def manual_run(self, used_genes=None):

        if used_genes == None:
            used_genes = self.genes_perm

        self.results_iteration = {k : [] for k,v in self.results_record.items()}


        self.i_genes, phase = self.phase_selection(self.counter,  used_genes)
        self.i_genes = np.asarray([i for i in self.i_genes if i not in self.genes_perm])

        self.current_genes += 1

        for j in tqdm.tqdm(range(self.i_genes.shape[0])):

            i_gene = self.i_genes[j]

            if self.counter >= 1:
                i_gene = np.concatenate((np.array(self.genes_perm).reshape(-1), np.array([i_gene])))
            if self.counter == 0:
                i_gene = [self.i_gene, i_gene]

            self.results_iteration['idx'].append(i_gene)
            self.results_iteration['genes'].append(i_gene)


            self.run_model(i_gene, self.X_data, self.Y_data, 'foward')


        # self.genes_perm = self.results_iteration['idx']
        self.genes_perm = self.results_iteration['idx'][self.results_iteration['train_error'].index(min(self.results_iteration['train_error']))]

        # self.genes_perm = self.results_iteration['idx']
        self.base_score = self.results_iteration['train_error'][self.results_iteration['train_error'].index(min(self.results_iteration['train_error']))]


        self.counter += 1

        return self.genes_perm, self.base_score

    def manual_reverse(self):
        remove = True
        remove_count = 0
        used_genes = self.genes_perm

        while remove == True and len(self.genes_perm) > 3:

            # len(self.genes_perm) makes sure the previously added gene isn't removed
            for m in range(0, len(self.genes_perm) - 1):
                self.results_remove = {k: [] for k, v in self.results_record.items()}

                gene_remove = self.genes_perm.copy()
                gene_remove = np.delete(gene_remove, m)
                remove_count += 1

                self.results_remove['idx'].append(gene_remove)
                self.results_iteration['genes'].append(gene_remove)
                # run model each time with a gene removed
                self.run_model(gene_remove, self.X_data, self.Y_data, type='reverse')

                # if a new result is better than the baseline error - let the gene be removed
                if self.results_remove['train_error'] < self.base_score:
                    self.base_score = self.results_remove['train_error']

                    self.genes_perm = gene_remove
                    self.current_genes -= 1
                    # if a gene is removed, the loop restarts
                    break

                if remove_count >= len(self.genes_perm)-1:
                    remove = False
                    break

        else:
            #print('Genes: ')
            # for j in self.genes_perm:
            #     print(j)

            #print('\n Best error: ', self.base_score)
            return self.genes_perm, self.base_score




    def status_update(self):
        print('Gene count: {} \n'.format(self.counter+1))
        print('Best genes: ')
        for j in self.genes_perm:
            print(j)
        print('\n Best error', self.base_score)


        counts = self.counts.copy()



        used_genes = self.rhythmic_scores.loc[self.genes_perm]['phase_bin']

        # make sure genes aren't repeated
        used_genes = used_genes.loc[np.unique(used_genes.index)]


        for j in used_genes.values:
            counts[j] += 1


