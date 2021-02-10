import tensorflow as tf
import numpy as np
from baseconvert import base

# From Tensorflow Tutorial
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

# Originally from tensorflow tutorial
def create_padding_mask(seq):
  seq = tf.cast(tf.math.not_equal(tf.math.reduce_sum(seq, axis=-1), 0), tf.float32)

  return seq # Output 1's where attention is given, and 0's for masked parts of a sequence... this is opposite of the tutorial

# From Tensorflow Tutorial
# positions = total number of positions for each word
# d_model is the depth of the embedding vector
def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# Discriminator Layer
def discriminator_layer(neurons, window_size, output_dims):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(neurons, activation=tf.nn.swish),
            tf.keras.layers.Dense(1),
        ]
    )


def calc_kmer_numeric_tuple(k, n):
    x = list(base(int(n), 10, 5))
    x = [0] * (k - len(x)) + x
    return x[:k]


NUMERIC_TO_STRING = {
    0: "A",
    1: "T",
    2: "N",
    3: "C",
    4: "G",
}


def convert_tuple_to_string(n):
    x = ""
    for i in n:
        x += NUMERIC_TO_STRING.get(i)
    return x


def convert_tuple_to_np(k, x):
    out = np.zeros((k, 5))
    for i in range(k):
        if x[i] > 4:
            x[i] = 0
        if x[i] < 0:
            x[i] = 0
        out[i][x[i]] = 1
    return np.reshape(out, (k * 5))


def convert_string_to_nparray(s):
    d = list()
    for x in s:
        if x == "A":
            d.append(0)
            continue
        if x == "T":
            d.append(1)
            continue
        if x == "C":
            d.append(3)
            continue
        if x == "G":
            d.append(4)
            continue
        if x == "N":
            d.append(2)
            continue
    return np.array(d)


def convert_string_to_nparray_tuple(k, s):
    out = np.zeros(k * 5)
    for x in range(k):
        if s[x] == "A":
            out[5 * x] = 1
        if s[x] == "T":
            out[5 * x + 1] = 1
        if s[x] == "C":
            out[5 * x + 3] = 1
        if s[x] == "G":
            out[5 * x + 4] = 1
        if s[x] == "N":
            out[5 * x + 2] = 1
    return out
