
# Import libraries
import os, sys, cv2, matplotlib.pyplot as plt, numpy as np, shutil, itertools, pickle, pandas as pd, seaborn as sn, math, time
from random import seed, random, randint
from scipy.spatial import distance
import random
import tensorflow as tf
from keras import backend as K
from keras.models import Model, load_model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Embedding, AveragePooling1D, dot, UpSampling2D, concatenate, BatchNormalization, LSTM, Multiply, Conv2D, MaxPool2D, Add, dot, GlobalMaxPool1D, Dropout, Masking, Activation, MaxPool1D, Conv1D, Flatten, TimeDistributed, Lambda
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.regularizers import l2

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from mpl_toolkits.mplot3d import Axes3D

import librosa 
import librosa.display

import soundfile as sf

emb_size = 32
alpha = 0.8


# Triplet loss
def triplet_loss(y_true, y_pred):

    anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
    distance1 = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    distance2 = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    loss = tf.reduce_mean(tf.maximum(distance1 - distance2 + alpha, 0))

    return loss


# Build Model

def get_model(r,c) :

    # hyper-parameters
    n_filters = 64
    filter_width = 3
    dilation_rates = [2**i for i in range(8)] 

    history_seq = Input(shape=(r, c))
    x = history_seq

    skips = []
    count = 0
    for dilation_rate in dilation_rates:
        x = Conv1D(filters=n_filters,
                    kernel_size=filter_width, 
                    padding='causal',
                    dilation_rate=dilation_rate, activation='relu', name="conv1d_dilation_"+str(dilation_rate))(x)
        
        x = BatchNormalization()(x)


    out = Conv1D(32, 16, padding='same')(x)
    out = BatchNormalization()(out)
    out = Activation('tanh')(out)
    out = GlobalMaxPool1D()(out)

    model = Model(history_seq, out)
    model.compile(loss='mse', optimizer='adam')

    input1 = Input((r,c), name="Anchor Input")
    input2 = Input((r,c), name="Positive Input")
    input3 = Input((r,c), name="Negative Input")

    anchor = model(input1)
    positive = model(input2)
    negative = model(input3)


    concat = concatenate([anchor, positive, negative], axis=1)

    siamese = Model([input1, input2, input3], concat)


    siamese.compile(optimizer='adam', loss=triplet_loss)

    print(siamese.summary())

    return model, siamese

