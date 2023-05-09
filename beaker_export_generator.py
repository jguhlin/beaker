# 12 Oct model
# Less layers!
# First layer can attend to self, the rest of the layers mask the self word...

import os

# os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

# from tensorflow.keras.mixed_precision import experimental as mixed_precision

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

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
from biobeaker.utils import get_angles, positional_encoding
from biobeaker import BEAKER
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
    activation=tfa.activations.mish,
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

# Define the model
batch_input = Input(
    shape=(2, window_size + 1, k * 5), dtype="float32", name="BatchInput"
)

contexts_a = magic(batch_input[:, 0])
contexts_b = magic(batch_input[:, 1])
enc_outputs_a, _, _ = transformer(contexts_a, True)
enc_outputs_b, _, _ = transformer(contexts_b, True)


def matched_layer():
    return tf.keras.Sequential(
        [
            # tf.keras.layers.Dense(neurons, activation=tfa.activations.mish),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="Matched",
    )


def rc_layer(neurons):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(neurons, activation=tfa.activations.mish),
            tf.keras.layers.Dense(neurons, activation=tfa.activations.mish),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="Rc",
    )


def discriminator_layer(neurons):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(neurons, activation=tfa.activations.mish),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="Discriminator",
    )


Matched = matched_layer()
DropMatched = Dropout(dropout_rate)

Rc = rc_layer(2048)
DropRc = Dropout(dropout_rate)

Discriminator = discriminator_layer(2048)
DropDiscriminator = Dropout(dropout_rate)

CosSim = tf.keras.layers.Dot(axes=-1, normalize=True)

# TODO: I think this is only looking at the CLS token...
# out1 = Matched(DropMatched(tf.concat([enc_outputs_b[:, 0], enc_outputs_a[:, 0]], axis=-1)))
out1 = Matched(CosSim([enc_outputs_a[:, 0], enc_outputs_b[:, 0]]))
out2 = Rc(DropRc(tf.concat([out1, enc_outputs_b[:, 0], enc_outputs_a[:, 0]], axis=-1)))

out0a = tf.squeeze(Discriminator(DropDiscriminator(enc_outputs_a[:, 1:])), name="Dis0")
out0b = tf.squeeze(Discriminator(DropDiscriminator(enc_outputs_b[:, 1:])), name="Dis1")

model = Model(inputs=[batch_input], outputs=[out0a, out0b, out1, out2])

# Load up the weights
weights = np.load(
    "weights/weights_wide_singlelayer_k21_3Aug2020model_21_dims_32_epochs256.npy",
    allow_pickle=True,
)
magic.set_weights([weights[0][0]])

print(model.summary())

latest = tf.train.latest_checkpoint("beaker_medium_nt_triple2023/")
if latest:
    print("Loading checkpoint")
    print(latest)
    model.load_weights(latest).expect_partial()
    print("Checkpoint loaded")
else:
    print("Checkpoint NOT loaded")

transformer.save_weights("beaker_medium_tripleloss")
Matched.save_weights("matched_layer_beaker_medium_tripleloss")
