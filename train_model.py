import sys
import pandas as pd
import numpy as np
import math
import pickle
from sklearn.preprocessing import normalize, StandardScaler
import random
import os
from numpy.linalg import norm
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import argparse

#import the neural network model
from chronogauge.model_nn import MultiOutputNN
from chronogauge.utils import cyclic_time, time24, errors


#make sure same seed is used throughout
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)




#these genes are 17 clock genes that will be used as default features if an ID is not provided
clock_genes = ['AT5G61380', 'AT2G46830', 'AT1G01060', 'AT5G02810', 'AT2G46790', 'AT2G25930', 'AT1G22770', 'AT5G42900',
       'AT3G46640','AT5G59570', 'AT5G60100', 'AT3G22380', 'AT4G39620', 'AT5G57360', 'AT2G31870',  'AT2G21070',
       'AT3G20810']

parser = argparse.ArgumentParser(description='An example script which can train, save and test a NN model for CT prediction')
parser.add_argument('--model_id', type=int, help='The numeric ID of a model within the tuned ensemble of NN models. If None, will use default parameters and all gene features from the ensemble.')
parser.add_argument('--x_training', type=str, help='Define the training expression matrix .csv file. If None, will use defualt training data.', default='data/expression_matrices/x_training.csv')
parser.add_argument('--target_training', type=str, help='Define model targets - sampling times in hours for training data across 1st column of .csv file. If None, use default training targets.', default='data/targets/target_training.csv')
parser.add_argument('--x_test', type=str, help='Define the test expression matrix .csv file. If None, will not test.')
parser.add_argument('--target_test', type=str, help='Sampling times in hours for test data across 1st column of .csv file.\nIf None, will report CT predictions but not errors.')
parser.add_argument('--out_results', type=str, help='Directory to output test results in .csv format. If None, will output to results/test_results directory.', default='results/test_results')
parser.add_argument('--out_model', type=str, help='Directory to output a trained model in .h5 format. If None, will not output model.')
parser.add_argument('--scale_test_independently', type=int,choices=[0, 1], help='Determines whether test data is scaled independendly of the training. 0 for non-independent, 1 for independent. Default 0, non-independent.\nIndependent scaling is essential when the test data is generated from non RNA-seq data. Requires at least 2 time-points ~12 hours apart.',
                    default=0)

args = parser.parse_args()



def initial_check():
    """
    Summarizes the inputs to the script in addition to the choice of model features & hyperparameters to fit.
    """
    
    #check to see if model ID is provided, if not it will run NN with default paramters and all 347 gene features
    if args.model_id == None:
        print('Model ID: no model ID defined. Running model with default parameters & 17 core clock genes.')
    else:
        print('Model ID: running model number {}'.format(args.model_id))

    #reports training data being used
    print('Training matrix: using {} to train model.'.format(args.x_training))

    #check to see if training targets are provided, if not it will attempt to use the columns of the training matrix as a target
    if args.target_training == None:
        print('Training sampling times: YOU MUST PROVIDE VALID TARGETS FOR TRAINING.'.format(args.x_training))
        sys.exit()
    else:
        print('Training sampling times: using {} as model targets.'.format(args.target_training))

    #check if model is being saved
    if args.out_model == None:
        print('Model saving: model will not be saved')
    else:
        print('Model saving: model saved in {}.'.format(args.out_model))
        if not os.path.exists(args.out_model):
            os.makedirs(args.out_model)


    #check whether there is a test dataset
    if args.x_test is None:
        print('No test matrix provided. Will not apply test.')

    else:
        print('Test matrix: using {} to test model.'.format(args.x_test))

        #check if sampling times are included for test data, if not will return just predictions
        if args.target_test is None:
            print('Test sampling times: No test sampling times provided. Reporting only CT predictions.'.format(args.x_training))
        
        # check whether test data will be scaled independelty of the training data or not.
        if args.scale_test_independently == 0:
            print('Test data will be z-score scaled with reference to the training data.')
        else:
            print('Test data will be z-score scaled independelty of the training data.')

        print('Results directory: Test results saved to {}.'.format(args.out_results))
        if not os.path.exists(args.out_results):
            os.makedirs(args.out_results)


def model_parameters(model_id=None):
    """
    Defines features & hyperparameters used by model.
    
    Parameters
    ----------
    model_id : int
        The numerical id of a previously defined model corresponding to a set of gene features and hyperparamters.
        If None, the 17 clock genes will be used with default hyperparamters instead.
        
    Returns
    ----------
    feature_set : np.ndarray
        Array of gene features the model will fit to.
    lr : float
        Learning rate used by the model.
    l2 : float
        L2 regularization factor used by the model. NOTE we use lr for l2, as this was emperically found to be an optimal solution.
    batches : int
        Batch size used by the model.
    """
    
    # use core clock genes with default paramters
    if model_id is None:
        l2 = 0.000001
        batches = 1
        lr = 0.000001
        
        feature_set = clock_genes

    # load and select model parameters for specific
    # note, model tuning mistakenly used learning rate values flor l2 regularization. While there is no theoretical basis for this, the models emprically work well based on cross-validation results
    else:
        with open('data/model_parameters/model_parameters.p', 'rb') as fin:
            params = pickle.load(fin)
        params = params[model_id]
        lr = params['lr']
        batches = params['batches']
        l2 = params['lr']
    
        # select specific feature set for this model
        all_gene_features = pd.read_csv('data/model_parameters/gene_features_unadjusted.csv', index_col=0)
        feature_set = all_gene_features.iloc[model_id].dropna().to_numpy()
    
    return feature_set, lr, l2, batches


def load_data(x_data, y_data, features):
    """
    Loads expression data and sampling time targets used for either model training or testing.

    Parameters
    ----------
    x_data : str
        The file path for the CSV file containing a gene expression matrix
    y_data : str
        The file path for the CSV file containing the target labels. Required for model training.
        If None, no target labels will be loaded and processed targets will not be returned
    features : np.ndarray
        Array of gene features used by the model
        
    Returns
    ----------
    x_data_features : pd.DataFrame
        The expression matrix cotaining only defined gene features where rows are samples and columns are features.
    y_times : np.ndarray or None
         The target labels as hourly time-points in the 24-hour day, or None if y_data is None.
    y_circular : np.ndarray or None
        The target labels transformered as circular cosine and sine values, or None if y_data is None.
    """
    # load expression matrix
    x_data = pd.read_csv(x_data, index_col=0).T
    
    #ensure x_data columns & features are consistent
    features = np.intersect1d(x_data.columns, features)
    x_data = x_data[features]
    
    if y_data is None:
        return x_data
    
    else:
        y_data = pd.read_csv(y_data, index_col=0).iloc[:,0]
        y_data = np.asarray([float(i) % 24 for i in y_data])
        y_times = y_data.copy()
        y_circular = cyclic_time(y_data)
        
        return x_data, y_times, y_circular

def process_outputs(model_outputs, sample_labels, sampling_times=None):
    """
    Generates a dataframe that summuarizes the model's prediction for each sample in the test data.

    Parameters
    ----------
    model_output : np.ndarray
        Matrix repsenting the model time prediction as circular sine and cosine values.
    sample_labels : np.ndarray
        Array of string labels corresponding to each row of the model_output matrix. For practically, 
        we use the index of the test expression dataframe.
    sampling_times : np.ndarray or None
        Array of float times corresponding to each row of the model_output matrix used to calculate error.
        If None, error will not be calculated and results_predictions will include no error metrics.

    Returns
    ----------
    results_predictions : pd.DataFrame
        A dataframe with the following strcuture:
        
        - Always included Columns:
            - 'CT estimate (hr)' : float
                The predicted time of each sample in hours within a 24-hour modulus.
                
        - Conditionally included columns (if sampling_times is not None):
            - 'Sampling time (hr)' : float 
                The true time each sample was harvested at.
            - 'Error (mins)' : float
                The error of each sample's prediction relative to the true sampling time.
            - 'Absolute error (mins)': float
                The absolute error of each sample's prediction relative to the true sampling time.
                
        - Rows:
            Each row of the dataframe corresponds to a sample inputted into the model. 
    """
    #convert sine & cosine values into real time
    test_preds = time24(model_outputs)
    results_predictions = pd.DataFrame(index=sample_labels, data=test_preds, columns=['CT estimate (hr)'])
    
    if sampling_times is not None:
        results_predictions['Sampling time (hr)'] = sampling_times
        results_predictions['Error (mins)'] = errors(results_predictions['CT estimate (hr)'],
                                                     results_predictions['Sampling time (hr)'])
        results_predictions['Absolute error (mins)'] = np.absolute(results_predictions['Error (mins)'])
    
    return results_predictions


def main():
    # summarize the inputs
    initial_check()

    # set the model you wish to train
    
    # if model ID isn't defined, just use default params and 17 clock gene features
    if args.model_id is None:
        model_number = 'DEFAULT'
        feature_set, lr_val, l2_val, n_batch = model_parameters()
        
    # if model ID defined
    else:
        print(args.model_id)
        model_number = int(args.model_id)

        feature_set, lr_val, l2_val, n_batch = model_parameters(model_number)

    # load training data & targets
    print('Loading training data.')
    X_train, Y_time, Y_train = load_data(args.x_training, args.target_training, feature_set) 
    feature_set = X_train.columns
    

    # check if test data is None
    if args.x_test is not None:
        # check if test targets is None
        if args.target_test is None:
            # load test data
            X_test = load_data(x_data=args.x_training, features=feature_set)
        else:
            # load test data with targets. Circular values not necessary
            X_test, Y_test, _ = load_data(args.x_test, args.target_test, feature_set)
    
        #ensure features across X_train and X_test are consistent
        feature_set = np.intersect1d(X_train.columns, X_test.columns)
        X_train = X_train[feature_set]
        X_test = X_test[feature_set]
    
    # standardize training data using z-score scaling
    print('Standardize training data.')
    scaler = StandardScaler()
    X_train = pd.DataFrame(data=scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    
    
    # define the model using seleceted hyperparameters
    model = MultiOutputNN(learning_rate=lr_val, l2_reg=l2_val, batch_size=n_batch,
                          SEED=SEED).nn_model()
    # this stops the model from training when loss no longer decreases
    early_stop = EarlyStopping(patience=25, restore_best_weights=True, monitor='loss', mode='min')
    
    print('Fitting model to training data.')
    model.fit(X_train.astype('float32'), Y_train.astype('float32'),
              batch_size=n_batch, epochs=1000, verbose=False, callbacks=[early_stop])

    # model can be saved or applied directly
    if args.out_model is not None:
        print('Saving model to: {}/model_{}.h5'.format(args.out_model, model_number))
        model.save(args.out_model + '/model_{}.h5'.format(model_number))
    
    # if using test data, standardize test data
    if args.x_test is not None:
        if args.scale_test_independently == 0:
            print('Scaling test data using training data scaling factor.')
            X_test = pd.DataFrame(data=scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
        else:
            print('Scaling test data idepedently of the training data.')
            X_test = pd.DataFrame(data=scaler.fit_transform(X_test), index=X_test.index, columns=X_test.columns)
    



    # make predictions for the test data
    if args.x_test is not None:
        print('Make predictions on test data')
        test_preds = normalize(model(X_test.values.astype('float32')))
        
        # get results dataframe
        test_results = process_outputs(test_preds, np.asarray(X_test.index), Y_test)
        
        output_file = args.x_test.split('/')[-1].split('.')[0] + '_{}_results.csv'.format(model_number)
        print('Saving test results to: ' + args.out_results+'/'+output_file)
        test_results.to_csv(args.out_results+'/'+output_file)
    
    
if __name__ == "__main__":
    main()
