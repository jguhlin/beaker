# Model settings
# Number of dimensions we are encoding the vector as
# Alphabet size 
# Support 24 amino acids
# including X 
ALPHABET_SIZE = 23
k = 11
dims = 16

## No more manual settings here!
kmer_space = ALPHABET_SIZE**k
# Set the model parameters here

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Imports
import tensorflow as tf# Sequence object for Keras
class KMERSeqs(tf.keras.utils.Sequence):
    def __init__(self, k, batch_size):
        self.k = k
        self.kmer_space = 5**k
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(self.kmer_space/ self.batch_size))

    def __getitem__(self, idx):
        self.kmers = [randint(0, self.kmer_space) for x in range(int(self.batch_size))]
        self.kmers_processed = [calc_kmer_numeric_tuple(self.k, x) for x in self.kmers]
        self.kmers = self.kmers_processed
        
        batch = [np.array(convert_tuple_to_np(k, x)) for x in self.kmers]

        batch_x = np.asarray(batch)
        batch_y = batch_x
        
        return (batch_x, batch_y)


import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Embedding, Flatten, Lambda, Subtract, Input, Concatenate, AveragePooling1D, LocallyConnected1D, Conv1D, GaussianNoise
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import Sequence
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

import pandas as pd
import numpy as np
import itertools
import math

from random import shuffle, randrange, randint, choices, random
from numba import jit, njit
from numba.typed import List, Dict
from functools import partial
from baseconvert import base

from lib.useful import calc_kmer_numeric_tuple, convert_tuple_to_string, calc_distance, convert_tuple_to_np, cos_sim, convert_string_to_nparray, convert_string_to_nparray_tuple

# Using pam250 here..

import Bio.SubsMat as subsmat
import Bio.SubsMat.MatrixInfo as mi
pam250 = mi.pam250
a = list(set([x[0] for x in mi.pam250.keys()]))
a.sort()
a
idx_to_aa = dict()
aa_to_idx = dict()
i = 0
for x in a:
    idx_to_aa[i] = x
    aa_to_idx[x] = i
   
    i = i + 1
    
pam250_full = Dict()
for x, y in pam250.items():
    pam250_full[x] = y
    kr = list(x)
    kr.reverse()
    pam250_full[tuple(kr)] = y
    

def convert_tuple_to_np(k, x):
    out = np.zeros((k,ALPHABET_SIZE))
    for i in range(k):
        out[i][x[i]] = 1
    return out

# @njit
def calc_kmer_numeric_tuple(k, n) -> List:
    x = list(base(int(n), 10, ALPHABET_SIZE))
    x = [0] * (k - len(x)) + x
    result = List()
    result.extend(x)
    return x

def calc_score(k, z) -> float:
    score = 0.0

    x = z[0]
    y = z[1]
    
    for i in range(k):
        xaa = idx_to_aa[x[i]]
        yaa = idx_to_aa[y[i]]
        score = score + pam250_full[(xaa, yaa)]
    
    return score

def calc_normalized_score(k, z) -> float:
    x = z[0]
    y = z[1]
    max_score = max(calc_score(k, (x, x)), calc_score(k, (y, y)))
    score = calc_score(k, (x, y))
    nscore = score/max_score
    if nscore < 0:
        return 0
    else:
        return nscore

class KMERProtSeqs(tf.keras.utils.Sequence):
    def __init__(self, k, vocab_size):
        self.k = k
        self.kmer_space = ALPHABET_SIZE**k
        self.vocab_size = vocab_size
        self.kmers = [randint(0, self.kmer_space) for x in range(int(vocab_size))]
        self.kmers = [calc_kmer_numeric_tuple(self.k, x) for x in self.kmers]
       
    def __len__(self):
        return int(np.ceil(10000000 / float(self.vocab_size)))

    def __getitem__(self, idx):
        batch = list()
        new_kmers = list()
        
        for kmer in self.kmers:
            for _ in range(int(self.vocab_size)):
                kmerx = kmer.copy()
                for _a in range(randint(0, int(k/2))):
                    x = randint(0, k-1)
                    kmerx[x] = randint(0, ALPHABET_SIZE-1)

                kmery = kmer.copy()
                for _b in range(randint(0, int(k))):
                    x = randint(0, k-1)
                    kmery[x] = randint(0, ALPHABET_SIZE-1)

            batch.append([kmer,  kmerx])
            batch.append([kmer,  kmery])
            batch.append([kmerx, kmery])
            new_kmers.append(kmerx)
            new_kmers.append(kmery)
            
        self.kmers.extend(new_kmers)
        total_kmers = len(self.kmers)
        for _ in range(int(np.ceil(total_kmers*10))):
            x = randint(0, total_kmers-1)
            y = randint(0, total_kmers-1)
            batch.append([self.kmers[x], self.kmers[y]])
            
        replace_kmers = list()
        for _ in range(self.vocab_size):
            x = randint(0, total_kmers-1)
            replace_kmers.append(self.kmers[x])
        
        self.kmers = replace_kmers

        batch = np.array(batch)
        
        batch_y = list();
        batch_x = list()
        
        for x in batch:
            score = calc_distance(k, x)
#            if score > 0:
#                converted_score = -(score*2)+1
#                if converted_score > 0:
            batch_y.append(score)
            batch_x.append([convert_tuple_to_np(k, x[0]), convert_tuple_to_np(k, x[1])])
        
        # batch_y = [ for x in batch]
        # batch_x = np.asarray([[convert_tuple_to_np(k, x[0]), convert_tuple_to_np(k, x[1])] for x in batch])
        
        batch_x = np.asarray(batch_x)
                
#        return [np.array(batch_x[:,0]), np.array(batch_x[:,1])], np.asarray(batch_y)
        return [np.array(batch_x[:,0]), np.array(batch_x[:,1])],[np.asarray(batch_y), batch_x[:,0], batch_x[:,1]]

    
# Original model from get_weights, but now with KMERSeqs corrected....
# loss 0.27.. w/ Adam, dims=12
# After ~50 epochs, 100 steps per epoch, KMERSeqs(k, 256)


def model1():
    input1 = Input(shape=(k,ALPHABET_SIZE), dtype='float32', name="k1")
    input2 = Input(shape=(k,ALPHABET_SIZE), dtype='float32', name="k2")

    input1_flat = Flatten()(input1)
    input2_flat = Flatten()(input2)

    magic = Dense(dims,
                  use_bias=False,
                  activation=tf.nn.swish,
                  name="Magic",
                  kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=1),)

    k1m = magic(input1_flat)
    k2m = magic(input2_flat)

    subtracted = Subtract()([k1m, k2m])
    abs = tf.math.abs(subtracted)
    output = tf.keras.backend.sum(abs, axis=1)

    reverso = Dense(k*ALPHABET_SIZE*dims, use_bias=False, activation=tf.nn.swish, name="Reverso")
    reverso_output = Dense(k*ALPHABET_SIZE, name="ReversoOutput")
    reshaped = tf.keras.layers.Reshape((k, ALPHABET_SIZE))

    k1r = keras.activations.softmax(reshaped(reverso_output(reverso(k1m))), axis=2)
    k2r = keras.activations.softmax(reshaped(reverso_output(reverso(k2m))), axis=2)

#    output = tf.keras.layers.Dot(1, normalize=True)([k1m, k2m])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.1,
        decay_steps=1,
        decay_rate=0.99,
        staircase=False)

    model = Model(inputs=[input1, input2], outputs=[output, k1r, k2r])
    #model.compile(loss="mae", optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule))
    # model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))
    # model.compile(loss="mae", optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0))
    model.compile(loss="mae", optimizer=tf.keras.optimizers.Nadam())
    return model

modelckpt = keras.callbacks.ModelCheckpoint(
                filepath='proteindist_{epoch}',
                save_best_only=True,
                monitor='val_loss',
                verbose=1)

csvlogger = tf.keras.callbacks.CSVLogger("kmer_protein_encoder.log")

model = model1()

print(model.summary())

cur_epoch = 0
epochs = 512
model.fit(KMERProtSeqs(k, 256),
          initial_epoch=cur_epoch,
          validation_data=KMERProtSeqs(k, 8192).__getitem__(0),
          epochs=cur_epoch+epochs,
          steps_per_epoch=128,
          verbose=1,
          shuffle=True,
          callbacks=[csvlogger])

weights = model.get_weights()[0]
np.save("weights/nadam_weights_protein_k" + str(k)
    + "_dims_" + str(dims),
    np.array([weights]))
