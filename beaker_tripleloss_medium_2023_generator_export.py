print("Don't forget: set -x LD_LIBRARY_PATH $CONDA_PREFIX/lib/ ")

# 12 Oct model

import os

from biobeaker.utils import get_angles, positional_encoding
from biobeaker import BEAKER

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import time
import pyracular

from lib.useful import (
    calc_kmer_numeric_tuple,
    convert_tuple_to_string,
    calc_distance,
    convert_tuple_to_np,
    cos_sim,
    convert_string_to_nparray,
    convert_string_to_nparray_tuple,
)
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
    Reshape,
    GlobalAveragePooling1D,
    Dropout,
)
from tensorflow.keras.models import Model, Sequential

# Hyper parameters
k = 21
window_size = 32  # up to 511
num_layers = 8
embedding_dims = 32
output_dims = 128  # Output dims are also internal dims!
intermediate_dims = 256
num_heads = 8
dropout_rate = 0.15
max_positions = 512
batch_size = 64

transformer = BEAKER(
    num_layers,
    embedding_dims,
    output_dims,
    num_heads,
    intermediate_dims,
    max_positions,
    dropout=dropout_rate,
    attention_dropout=dropout_rate,
    activation=tfa.activations.gelu,
)

generator = BEAKER(
    6,
    embedding_dims,
    output_dims,
    8,
    256,
    max_positions,
    dropout=0.15,
    attention_dropout=0.15,
    activation=tfa.activations.gelu,
)

def matched_layer():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="Matched",
    )


def rc_layer(neurons, activation="relu"):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(neurons, activation=activation),
            tf.keras.layers.Dense(neurons, activation=activation),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="Rc",
    )


def discriminator_layer(neurons, activation="relu"):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(neurons, activation=activation),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="Discriminator",
    )


magic = Dense(
    embedding_dims,
    activation=tf.nn.swish,
    name="Magic",
    use_bias=False,
    trainable=False,
    dtype=tf.float32,
)
EPOCHS = 12

cls = np.asarray([[1] * 105])
mask = np.asarray([[0] * 105])

# Define the model
# Input is 2 with mask (and CLS token)
# input is another 2 without mask (also with CLS token)
batch_input = Input(
    shape=(4, window_size + 1, k * 5), dtype="float32", name="BatchInput"
)

truth1 = Input(shape=window_size, dtype=tf.float32, name="Truth1")
truth2 = Input(shape=window_size, dtype=tf.float32, name="Truth2")

contexts_a = magic(batch_input[:, 0])
contexts_b = magic(batch_input[:, 1])

BackToEmbeddings = tf.keras.layers.Dense(embedding_dims)

contexts_a_true = magic(batch_input[:, 2])
contexts_b_true = magic(batch_input[:, 3])

# Generator - Train to replace mask token
generated_a, _, _ = generator(contexts_a, True)
generated_b, _, _ = generator(contexts_b, True)
generated_a = BackToEmbeddings(generated_a)
generated_b = BackToEmbeddings(generated_b)

enc_outputs_a, _, _ = transformer(generated_a, True)
enc_outputs_b, _, _ = transformer(generated_b, True)

Matched = matched_layer()

Rc = rc_layer(1024)
DropRc = Dropout(dropout_rate)

Discriminator = discriminator_layer(1024)
DropDiscriminator = Dropout(dropout_rate)

CosSim = tf.keras.layers.Dot(axes=-1, normalize=True)
CosSimNoise = tf.keras.layers.GaussianNoise(0.1)

# TODO: I think this is only looking at the CLS token...
# out1 = Matched(DropMatched(tf.concat([enc_outputs_b[:, 0], enc_outputs_a[:, 0]], axis=-1)))
out1 = Matched(CosSimNoise(CosSim([enc_outputs_a[:, 0], enc_outputs_b[:, 0]])))
out2 = Rc(DropRc(tf.concat([out1, enc_outputs_b[:, 0], enc_outputs_a[:, 0]], axis=-1)))

out0a = tf.squeeze(Discriminator(DropDiscriminator(enc_outputs_a[:, 1:])), name="Dis0")
out0b = tf.squeeze(Discriminator(DropDiscriminator(enc_outputs_b[:, 1:])), name="Dis1")

# Generator Loss

print(tf.shape(truth1[:, :, tf.newaxis]))
print(tf.shape(generated_a))

a_score = truth1[:, :, tf.newaxis] * (generated_a[:, 1:, :] - contexts_a_true[:, 1:, :])
b_score = truth2[:, :, tf.newaxis] * (generated_b[:, 1:, :] - contexts_b_true[:, 1:, :])

print(tf.shape(a_score))
print(tf.shape(b_score))

gen_loss_a = tf.math.reduce_sum(tf.math.square(a_score), axis=-1)
gen_loss_b = tf.math.reduce_sum(tf.math.square(b_score), axis=-1)

#gen_loss_a = tf.math.reduce_sum(tf.math.square(generated_a - contexts_a_true), axis=-1)
#gen_loss_b = tf.math.reduce_sum(tf.math.square(generated_b - contexts_b_true), axis=-1)

model = Model(
    inputs=[batch_input, truth1, truth2], outputs=[out0a, out0b, out1, out2, gen_loss_a, gen_loss_b]
)

# Load up the weights
weights = np.load(
    "weights/weights_wide_singlelayer_k21_3Aug2020model_21_dims_32_epochs256.npy",
    allow_pickle=True,
)
magic.set_weights([weights[0][0]])

checkpoint_path = "beaker_medium_nt_triple2023_generator/model_{epoch:04d}.ckpt"

latest = tf.train.latest_checkpoint("beaker_medium_nt_triple2023_generator/")
if latest:
    print("Loading checkpoint")
    print(latest)
    model.load_weights(latest).expect_partial()
    print("Checkpoint loaded")
else:
    print("Checkpoint NOT loaded")

transformer.save_weights("beaker_medium_nt_triple2023_generator_transformer")
generator.save_weights("beaker_medium_nt_triple2023_generator_generator")
# BackToEmbeddings.save_weights("back_to_embeddings")
np.save("BackToEmbeddings.weights.npy", BackToEmbeddings.get_weights())
