import sys

# Set the model parameters here
# Number of dimensions we are encoding the vector as
k = int(sys.argv[1])
dims = int(sys.argv[2])
batch_size = 1024

## No more manual settings here!
kmer_space = 23**k

# Imports
import numpy as np
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
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from beaker_kmer_generator import KmerGenerator as kmer_generator

kg = kmer_generator()
kg.set_aa()
kg.set_threads( 16)
kg.set_k(k)
kg.set_seed(42)
kg.start()

vkg = kmer_generator()
vkg.set_aa()
vkg.set_threads(8)
vkg.set_k(k)
vkg.set_seed(1010)
vkg.start()


def gen():
    while True:
        data = kg.generate_pairs()
        for i in data:
            (k1a, k2a, score) = i
            yield (k1a, k2a), (score, k1a, k2a)


def vgen():
    while True:
        vdata = vkg.generate_pairs()
        for i in vdata:
            # (k1a, k2a, k1, k2, score, target_score) = i
            (k1a, k2a, score) = i
            yield (k1a, k2a), (score, k1a, k2a)


ds = (
    tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(2, k*23), dtype=tf.int32),
            (
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(k*23), dtype=tf.int32),
                tf.TensorSpec(shape=(k*23), dtype=tf.int32),
            ),
        ),
    )
    .batch(batch_size)
    .prefetch(128)
)

vds = (
    tf.data.Dataset.from_generator(
        vgen,
        output_signature=(
            tf.TensorSpec(shape=(2, k*23), dtype=tf.int32),
            (
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(k*23), dtype=tf.int32),
                tf.TensorSpec(shape=(k*23), dtype=tf.int32),
            ),
        ),
    )
    .batch(batch_size)
    .prefetch(128)
)

def model2(opt):
    model_input = Input(shape=(2, k * 23), dtype="float32", name="kmers")
    input1_flat = model_input[:, 0, :]
    input2_flat = model_input[:, 1, :]

    magic = Dense(
        dims*4,
        activation="gelu",
        name="Magic",
        dtype="float32",
#        kernel_initializer=tf.keras.initializers.RandomNormal(
#            mean=0.0, stddev=0.05, seed=42
#        ),
    )

    magic2 = Dense(
        dims,
        activation="gelu",
        name="Magic2",
        dtype="float32",
#        kernel_initializer=tf.keras.initializers.RandomNormal(
#            mean=0.0, stddev=0.05, seed=42
#        ),
    )

    k1m = magic2(magic(input1_flat))
    k2m = magic2(magic(input2_flat))

    subtracted = Subtract()([k1m, k2m])
    abs = tf.math.abs(subtracted)
    output = tf.keras.backend.sum(abs, axis=1)

    reverso = Dense(
        k * 9 * 3 * dims, use_bias=False, activation=tf.nn.swish, name="Reverso"
    )
    reverso_output = Dense(k * 23, name="ReversoOutput")
    reshaped = tf.keras.layers.Reshape((k, 23))

    k1r = Flatten()(keras.activations.softmax(reshaped(reverso_output(reverso(k1m))), axis=2))
    k2r = Flatten()(keras.activations.softmax(reshaped(reverso_output(reverso(k2m))), axis=2))

    model = Model(inputs=[model_input], outputs=[output, k1r, k2r])
    model.compile(loss="mse", optimizer=opt)  # tf.keras.optimizers.Nadam())
    return model


opt = tf.keras.optimizers.Nadam()

model = model2(opt)
print(model.summary())
print("At first training step...")

opt.lr = 1e-6

print(opt.lr)

logcb = tf.keras.callbacks.CSVLogger(
    "log_aa_k_{}_dims_{}.csv".format(k,dims), separator=',', append=True
)


cur_epoch = 0
epochs = 8192
model.fit(
    ds,
    initial_epoch=cur_epoch,
    validation_data=vds,
    epochs=cur_epoch + epochs,
    steps_per_epoch=512,
    validation_steps=128,
    verbose=1,
    shuffle=False,
    callbacks = [logcb],
)

weights = model.get_weights()
np.save(
    "weights/weights_wide_singlelayer_aa_k{}_11Nov2022_swish_model_dims_{}_epochs_{}".format(
        k, dims, epochs
    ),
    weights,
)

cur_epoch = cur_epoch + epochs
epochs = 256
opt.lr = 0.0001

model.fit(
    ds,
    initial_epoch=cur_epoch,
    validation_data=vds,
    epochs=cur_epoch + epochs,
    steps_per_epoch=256,
    validation_steps=32,
    verbose=1,
    shuffle=False,
    callbacks = [logcb],
)
#          callbacks=[cb])

np.save(
    "weights/weights_wide_singlelayer_aa_k{}_11Nov2022_swish_model_dims_{}_epochs_{}".format(
        k, dims, epochs
    ),
    weights,
)

opt.lr = 0.00001
cur_epoch = cur_epoch + epochs
epochs = 256
model.fit(
    ds,
    initial_epoch=cur_epoch,
    validation_data=vds,
    epochs=cur_epoch + epochs,
    steps_per_epoch=256,
    validation_steps=32,
    verbose=1,
    shuffle=False,
    callbacks = [logcb],
)
#          callbacks=[cb])

weights = model.get_weights()[0]

np.save(
    "weights/weights_wide_singlelayer_aa_k{}_11Nov2022_swish_model_dims_{}_epochs_{}".format(
        k, dims, epochs
    ),
    weights,
)
