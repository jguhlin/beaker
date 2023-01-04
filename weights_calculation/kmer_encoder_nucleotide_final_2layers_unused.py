# Set the model parameters here
import os
import aim
from aim import Run
from aim.tensorflow import AimCallback

# Number of dimensions we are encoding the vector as
k = 21
dims = 32

## No more manual settings here!
kmer_space = 5**k

# New run
run = Run()
run["hparams"] = {
    "activation": "relu",

}

cb = AimCallback(repo='./kmer_encoder_log_dir', experiment='2layers')

# Imports
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense, Embedding, Flatten, Lambda, Subtract, Input, Concatenate, AveragePooling1D, LocallyConnected1D, Conv1D, GaussianNoise, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import Sequence
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from prefetch_generator import BackgroundGenerator, background,__doc__

import pandas as pd
import numpy as np
import itertools
import math

from random import shuffle, randrange, randint, choices, random
from numba import jit, njit
from numba.typed import List, Dict
from functools import partial
from baseconvert import base

from lib.useful import calc_kmer_numeric_tuple, convert_tuple_to_string, calc_distance_score, convert_tuple_to_np, cos_sim, convert_string_to_nparray, convert_string_to_nparray_tuple, shifty_score

def convert_tuple_to_np(k, x):
    out = np.zeros((k,5))
    for i in range(k):
        out[i][x[i]] = 1
    return out

# Sequence object for Keras
class KMERSeqs(tf.keras.utils.Sequence):
    def __init__(self, k, batch_size):
        self.k = k
        self.kmer_space = 5**k
        self.batch_size = batch_size
        self.kmers = [randint(0, self.kmer_space) for x in range(int(batch_size))]
        self.kmers = [calc_kmer_numeric_tuple(self.k, x) for x in self.kmers]
       
    def __len__(self):
        return int(np.ceil(10000000 / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = list()
        new_kmers = list()
        
        for kmer in self.kmers:
            for _ in range(int(self.batch_size)):
                kmerx = kmer.copy()
                for _a in range(randint(0, math.ceil(k/3))):
                    x = randint(0, k-1)
                    kmerx[x] = kmerx[x] + (1 if random() < 0.5 else -1)
                    if kmerx[x] > 4:
                        kmerx[x] = 0
                    elif kmerx[x] < 0:
                        kmerx[x] = 4
                    
                kmery = kmer.copy()
                for _b in range(randint(0, math.ceil(k/3))):
                    x = randint(0, k-1)
                    kmery[x] = kmery[x] + (1 if random() < 0.5 else -1)
                    if kmery[x] > 4:
                        kmery[x] = 0
                    elif kmery[x] < 0:
                        kmery[x] = 4

            if len(kmer) > k:
              print("Error with kmer: " + str(kmer))

            if len(kmery) > k:
              print("Error with kmery: " + str(kmery))

            if len(kmerx) > k:
              print("Error with kmerx: " + str(kmerx))

            batch.append([kmer,  kmerx])
            batch.append([kmer,  kmery])
            batch.append([kmerx, kmery])
            new_kmers.append(kmerx)
            new_kmers.append(kmery)
            
        self.kmers.extend(new_kmers)
        total_kmers = len(self.kmers)
        for _ in range(int(np.ceil(total_kmers*4))):
            x = randint(0, total_kmers-1)
            y = randint(0, total_kmers-1)
            batch.append([self.kmers[x], self.kmers[y]])
            
        replace_kmers = list()
        for _ in range(self.batch_size):
            x = randint(0, total_kmers-1)
            replace_kmers.append(self.kmers[x])
        
        self.kmers = replace_kmers

        shuffle(batch)
        batch = np.array(batch)

        batch_y = list()
        batch_x = list()
        
        for x in batch:
          dist = shifty_score(k, x)
          batch_y.append(dist)
          batch_x.append([convert_tuple_to_np(k, x[0]), convert_tuple_to_np(k, x[1])])
        batch_x = np.asarray(batch_x)
                
        return [np.array(batch_x[:,0]), np.array(batch_x[:,1])],[np.asarray(batch_y), batch_x[:,0], batch_x[:,1]]

def model2(opt):
    input1 = Input(shape=(k,5), dtype='float32', name="k1")
    input2 = Input(shape=(k,5), dtype='float32', name="k2")

    input1_flat = Flatten()(input1)
    input2_flat = Flatten()(input2)

    magic_inner = Dense(1024, activation=tf.nn.swish, name="MagicInner", use_bias=False, dtype="float32")

    magic = Dense(dims, activation="linear", name="Magic", use_bias=False, dtype="float32",
		kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.05, seed=42))

    k1m = magic(magic_inner(input1_flat))
    k2m = magic(magic_inner(input2_flat))

    subtracted = Subtract()([k1m, k2m])
    abs = tf.math.abs(subtracted)
    output = tf.keras.backend.sum(abs, axis=1)

    reverso = Dense(k*5*3*dims, use_bias=False, activation=tf.nn.swish, name="Reverso")
    reverso_output = Dense(k*5, name="ReversoOutput")
    reshaped = tf.keras.layers.Reshape((k, 5))

    k1r = keras.activations.softmax(reshaped(reverso_output(reverso(k1m))), axis=2)
    k2r = keras.activations.softmax(reshaped(reverso_output(reverso(k2m))), axis=2)

    model = Model(inputs=[input1, input2], outputs=[output, k1r, k2r])
    model.compile(loss="mse", optimizer=opt) # tf.keras.optimizers.Nadam())
    return model

opt = tf.keras.optimizers.Nadam()

model = model2(opt)
print(model.summary())
print("At first training step...")

validation_data = next(BackgroundGenerator(KMERSeqs(k, 8192)))

print(opt.lr)

cur_epoch = 0
epochs = 128
model.fit(BackgroundGenerator(KMERSeqs(k, 64)),
          initial_epoch=cur_epoch,
          validation_data=validation_data, #KMERSeqs(k, 8192).__getitem__(0),
          epochs=cur_epoch+epochs,
          steps_per_epoch=128,
          verbose=1, 
          shuffle=False,
          callbacks=[cb])

weights = model.get_weights()
np.save("weights/weights_wide_singlelayer_k21_17Nov_relu_model_" + str(k)
        + "_dims_" + str(dims) + "_epochs" + str(epochs),
    np.array([weights]))

cur_epoch=cur_epoch + epochs
epochs = 64
opt.lr = 0.0001

model.fit(KMERSeqs(k, 128),
          initial_epoch=cur_epoch,
          validation_data=KMERSeqs(k, 1024).__getitem__(0),
          epochs=cur_epoch+epochs,
          steps_per_epoch=128,
          verbose=1,
          shuffle=False,
          callbacks=[cb])

np.save("weights/weights_wide_singlelayer_k21_17Nov_relu_model_" + str(k)
        + "_dims_" + str(dims) + "_epochs" + str(epochs), np.array([weights]))

opt.lr = 0.00001
cur_epoch=cur_epoch + epochs
epochs = 64
model.fit(KMERSeqs(k, 256),
          initial_epoch=cur_epoch,
          validation_data=KMERSeqs(k, 1024).__getitem__(0),
          epochs=cur_epoch+epochs,
          steps_per_epoch=256,
          verbose=1,
          shuffle=False,
          callbacks=[cb])

weights = model.get_weights()[0]

np.save("weights/weights_wide_singlelayer_k21_17Nov_relu_model_" + str(k)
        + "_dims_" + str(dims) + "_epochs" + str(epochs), np.array([weights]))
