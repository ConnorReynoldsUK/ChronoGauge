import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, normalize, StandardScaler
import random
import os


#defines the angular loss function
def custom_loss(y_true, y_pred):
    return tf.reduce_mean((tf.math.acos(tf.reduce_sum((y_true * y_pred), axis=-1) / ((tf.norm(y_true, axis=1) * tf.norm(y_pred, axis=1)) + tf.keras.backend.epsilon()))))


class MultiOutputNN(object):
    def __init__(self, learning_rate=0.000001, l2_reg=0.00001, batch_size=1,
                 SEED=0, units_layer1=32, units_layer2=128, units_layer3=512):
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.SEED = SEED
        self.units_layer1 = units_layer1
        self.units_layer2 = units_layer2
        self.units_layer3 = units_layer3

        SEED = 0
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        os.environ['PYTHONHASHSEED'] = str(SEED)




    def nn_model(self):
    # create model
        model = Sequential()
        # relu
        model.add(Dense(self.units_layer1, kernel_initializer='normal',
                        activation='relu', kernel_regularizer=l2(self.l2_reg)))

        model.add(Dense(self.units_layer2, kernel_initializer='normal',
                        activation='relu', kernel_regularizer=l2(self.l2_reg)))

        model.add(Dense(self.units_layer3, kernel_initializer='normal',
                        activation='relu', kernel_regularizer=l2(self.l2_reg)))

        model.add(Dense(2, kernel_initializer='normal'))

        # Compile model
        model.compile(loss=custom_loss,
                      optimizer=Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False))
        return model
