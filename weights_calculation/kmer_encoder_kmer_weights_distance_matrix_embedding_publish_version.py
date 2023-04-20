# Ideas
# Be able to test multiple models
# Output as euc distance OR cos sim
# be able to test both
# Possible to have both euc distance AND cos sim? Prob not...


# Set the model parameters here

import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Number of dimensions we are encoding the vector as
k = 21
dims = 32

# Large vocab training set (higher learning rate)
# large_vocab_size = 15000
large_vocab_size = 15000

# Lower vocab training set (may improve accuracy? Still waiting to see...)
small_vocab_size = 4096
## No more manual settings here!
kmer_space = 5**k

# Imports
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    Flatten,
    Lambda,
    Subtract,
    Input,
    Concatenate,
    AveragePooling1D,
    LocallyConnected1D,
    Conv1D,
    GaussianNoise,
    BatchNormalization,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import Sequence
from tensorflow.keras import initializers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from prefetch_generator import BackgroundGenerator, background, __doc__

import pandas as pd
import numpy as np
import itertools
import math

from random import shuffle, randrange, randint, choices, random
from numba import jit, njit
from numba.typed import List, Dict
from functools import partial
from baseconvert import base

from lib.useful import (
    calc_kmer_numeric_tuple,
    convert_tuple_to_string,
    calc_distance_score,
    convert_tuple_to_np,
    cos_sim,
    convert_string_to_nparray,
    convert_string_to_nparray_tuple,
    shifty_score,
)


def convert_tuple_to_np(k, x):
    out = np.zeros((k, 5))
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
                for _a in range(randint(0, math.ceil(k / 3))):
                    x = randint(0, k - 1)
                    kmerx[x] = kmerx[x] + (1 if random() < 0.5 else -1)
                    if kmerx[x] > 4:
                        kmerx[x] = 0
                    elif kmerx[x] < 0:
                        kmerx[x] = 4

                kmery = kmer.copy()
                for _b in range(randint(0, math.ceil(k / 3))):
                    x = randint(0, k - 1)
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

            batch.append([kmer, kmerx])
            batch.append([kmer, kmery])
            batch.append([kmerx, kmery])
            new_kmers.append(kmerx)
            new_kmers.append(kmery)

        self.kmers.extend(new_kmers)
        total_kmers = len(self.kmers)
        for _ in range(int(np.ceil(total_kmers * 4))):
            x = randint(0, total_kmers - 1)
            y = randint(0, total_kmers - 1)
            batch.append([self.kmers[x], self.kmers[y]])

        replace_kmers = list()
        for _ in range(self.batch_size):
            x = randint(0, total_kmers - 1)
            replace_kmers.append(self.kmers[x])

        self.kmers = replace_kmers

        shuffle(batch)
        batch = np.array(batch)

        batch_y = list()
        batch_x = list()

        for x in batch:
            dist = shifty_score(k, x)
            #          if dist < 0.6:
            batch_y.append(dist)
            batch_x.append([convert_tuple_to_np(k, x[0]), convert_tuple_to_np(k, x[1])])
        batch_x = np.asarray(batch_x)
        #        batch_y = [-(calc_distance(k, x)*2)+1 for x in batch]
        #        batch_x = np.asarray([[convert_tuple_to_np(k, x[0]), convert_tuple_to_np(k, x[1])] for x in batch])

        return [np.array(batch_x[:, 0]), np.array(batch_x[:, 1])], [
            np.asarray(batch_y),
            batch_x[:, 0],
            batch_x[:, 1],
        ]


# 0.3259 with lr 0.05 and decay_steps=1
def model2():
    input1 = Input(shape=(k, 5), dtype="float32", name="k1")
    input2 = Input(shape=(k, 5), dtype="float32", name="k2")

    input1_flat = Flatten()(input1)
    input2_flat = Flatten()(input2)

    #    pruning_params = {
    #      'pruning_schedule': PolynomialDecay(0.0, 0.85, 32, 64),
    #    }

    # 8192 gets down to 1.3....
    # 2048 gets down to 1.4.....
    # 1024 gets down to about 2.5 I think... ?
    #  256 gets down to ~3.5
    #    magic = Dense(1024, activation=tf.nn.swish, name="Magic1", use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.1, seed=1))
    #    magic = prune_low_magnitude(magic, **pruning_params)

    magic = Dense(
        dims,
        activation=tf.nn.swish,
        name="Magic",
        use_bias=False,
        dtype="float32",
        #                kernel_regularizer=regularizers.l1(0.05),
        kernel_initializer=tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.005, seed=1
        ),
    )
    #    magic2 = prune_low_magnitude(magic2, **pruning_params)

    k1m = magic(input1_flat)
    k2m = magic(input2_flat)

    subtracted = Subtract()([k1m, k2m])
    abs = tf.math.abs(subtracted)
    output = tf.keras.backend.sum(abs, axis=1)

    reverso0 = Dense(
        k * 5 * 3 * dims, use_bias=False, activation=tf.nn.swish, name="Reverso0"
    )
    reverso1 = Dense(k * 5 * 3 * 8, use_bias=False, activation="relu", name="Reverso1")
    reverso_output = Dense(k * 5, name="ReversoOutput")
    reshaped = tf.keras.layers.Reshape((k, 5))

    k1r = keras.activations.softmax(
        reshaped(reverso_output(reverso1(reverso0(k1m)))), axis=2
    )
    k2r = keras.activations.softmax(
        reshaped(reverso_output(reverso1(reverso0(k2m)))), axis=2
    )

    #    squared = tf.keras.backend.pow(subtracted, 2)
    #    reshaped = tf.keras.layers.Reshape((dims, 1))(squared)
    #    summed = tf.keras.backend.sum(reshaped, axis=1)
    #    sqrt = tf.keras.backend.sqrt(summed)
    #    output = sqrt
    # 0.18.. so bad
    #    output = Dense(1, activation="exponential", use_bias=False)(tf.keras.layers.Dot(1, normalize=True)([k1m, k2m]))

    #    output = Dense(1, use_bias=False)(tf.keras.layers.Dot(1, normalize=True)([k1m, k2m]))
    # output = tf.keras.layers.Dot(1, normalize=True)([k1m, k2m])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.05, decay_steps=1, decay_rate=0.99, staircase=False
    )

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(0.05, 10000, 1e-5)

    model = Model(inputs=[input1, input2], outputs=[output, k1r, k2r])
    # model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))
    # model.compile(loss="mae", optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule))
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Nadam())
    #    model.compile(loss="mae", optimizer=tfa.optimizers.LAMB())
    return model


model = model2()
print(model.summary())

# large_training_set = KMERSeqs(k, 8192*10)
# large_validation_set = KMERSeqs(k, 8192*5)

# eval_data = KMERSeqs(k, 256).__getitem__(0)
# print(eval_data)

# print(model.predict(eval_data[0]))
# exit

print("At first training step...")

validation_data = next(BackgroundGenerator(KMERSeqs(k, 8192)))

csvlogger = tf.keras.callbacks.CSVLogger("kmer_encoder.log")

cur_epoch = 0
epochs = 512
model.fit(
    BackgroundGenerator(KMERSeqs(k, 64)),
    initial_epoch=cur_epoch,
    validation_data=validation_data,  # KMERSeqs(k, 8192).__getitem__(0),
    epochs=cur_epoch + epochs,
    steps_per_epoch=256,
    verbose=1,
    shuffle=False,
    callbacks=[csvlogger],
)
# callbacks=[UpdatePruningStep()])
# callbacks=[tensorboard_callback])


model.compile(loss="mse", optimizer=tf.keras.optimizers.Nadam(lr=0.01))
cur_epoch = 0
epochs = 512
model.fit(
    BackgroundGenerator(KMERSeqs(k, 64)),
    initial_epoch=cur_epoch,
    validation_data=validation_data,  # KMERSeqs(k, 8192).__getitem__(0),
    epochs=cur_epoch + epochs,
    steps_per_epoch=256,
    verbose=1,
    shuffle=False,
    callbacks=[csvlogger],
)
# callbacks=[UpdatePruningStep()])
# callbacks=[tensorboard_callback])

# model.save_weights('./checkpoints/kmerencodersparse_euc_32')

# pruned_model = strip_pruning(model)

# weights = pruned_model.get_weights()
# print(weights)
weights = model.get_weights()
np.save(
    "weights/weights_wide_singlelayer_k21_28Jan2021model_"
    + str(k)
    + "_dims_"
    + str(dims)
    + "_epochs"
    + str(epochs),
    np.array([weights]),
)

exit
