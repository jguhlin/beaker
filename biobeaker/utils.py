import tensorflow as tf
import numpy as np
 
# From Tensorflow Tutorial
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

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
