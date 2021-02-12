from __future__ import print_function
import sys
import matplotlib
import datetime
from tensorflow.examples.saved_model.integration_tests.mnist_util import INPUT_SHAPE
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras import layers
import tensorflow.keras as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.models as keras_models
from random import randrange
import Functions_RNN
import Generator_ads
import Memory_trace

def Build_Model():

    # 30_M_NORM_WO_REG_new : Model without Regularization new

    model = Sequential()
    model.add(layers.BatchNormalization( axis=-1,momentum=0.99, epsilon=0.001, center=True,trainable=True,
        scale=True, beta_initializer="zeros", gamma_initializer="ones", moving_mean_initializer="zeros",
        moving_variance_initializer="ones",renorm=False,renorm_momentum=0.99))

    model.add(Dense(200, activation='relu'))
    model.add(layers.BatchNormalization( axis=-1,momentum=0.99, epsilon=0.001, center=True,trainable=True,
        scale=True, beta_initializer="zeros", gamma_initializer="ones", moving_mean_initializer="zeros",
        moving_variance_initializer="ones",renorm=False,renorm_momentum=0.99))
    model.add(Dense(400, activation='sigmoid'))
    model.add(layers.BatchNormalization( axis=-1,momentum=0.99, epsilon=0.001, center=True,trainable=True,
        scale=True, beta_initializer="zeros", gamma_initializer="ones", moving_mean_initializer="zeros",
        moving_variance_initializer="ones",renorm=False,renorm_momentum=0.99))
    model.add(Dense(200, activation='sigmoid'))
    model.add(layers.BatchNormalization( axis=-1,momentum=0.99, epsilon=0.001, center=True,trainable=True,
        scale=True, beta_initializer="zeros", gamma_initializer="ones", moving_mean_initializer="zeros",
        moving_variance_initializer="ones",renorm=False,renorm_momentum=0.99))
    model.add(Dense(200, activation='relu'))
    model.add(layers.BatchNormalization( axis=-1,momentum=0.99, epsilon=0.001, center=True,trainable=True,
        scale=True, beta_initializer="zeros", gamma_initializer="ones", moving_mean_initializer="zeros",
        moving_variance_initializer="ones",renorm=False,renorm_momentum=0.99))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    return model