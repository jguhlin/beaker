import sys

# Set the model parameters here
# Number of dimensions we are encoding the vector as
k = int(sys.argv[1])
dims = int(sys.argv[2])
batch_size = 128
# activation = "relu"
# loss = "huber"
# bias = True

activation = str(sys.argv[3])
loss = str(sys.argv[4])
bias = bool(sys.argv[5])

## No more manual settings here!
kmer_space = 5**k

# Imports
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
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
kg.set_threads(4)
kg.set_k(k)
kg.set_seed(42)
kg.start()

vkg = kmer_generator()
vkg.set_threads(4)
vkg.set_k(k)
vkg.set_seed(1010)
vkg.start()


def gen():
    while True:
        data = kg.generate_pairs()
        for i in data:
            (k1a, k2a, score) = i
            yield (k1a, k2a), (score, np.reshape(k1a, (k, 5)), np.reshape(k2a, (k, 5)))


def vgen():
    while True:
        vdata = vkg.generate_pairs()
        for i in vdata:
            # (k1a, k2a, k1, k2, score, target_score) = i
            (k1a, k2a, score) = i
            yield (k1a, k2a), (score, np.reshape(k1a, (k, 5)), np.reshape(k2a, (k, 5)))


ds = (
    tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(2, k * 5), dtype=tf.int32),
            (
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(k, 5), dtype=tf.int32),
                tf.TensorSpec(shape=(k, 5), dtype=tf.int32),
            ),
        ),
    )
    .batch(batch_size)
    .prefetch(256)
)

vds = (
    tf.data.Dataset.from_generator(
        vgen,
        output_signature=(
            tf.TensorSpec(shape=(2, k * 5), dtype=tf.int32),
            (
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(k, 5), dtype=tf.int32),
                tf.TensorSpec(shape=(k, 5), dtype=tf.int32),
            ),
        ),
    )
    .batch(batch_size)
    .prefetch(256)
)


def model2(opt):
    model_input = Input(shape=(2, k * 5), dtype="float32", name="kmers")
    input1_flat = model_input[:, 0, :]
    input2_flat = model_input[:, 1, :]

    magic = Dense(
        dims,
        activation=activation,
        name="Magic",
        use_bias=bias,
        dtype="float32",
        kernel_initializer=tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.05, seed=42
        ),
    )

    k1m = magic(input1_flat)
    k2m = magic(input2_flat)

    subtracted = Subtract()([k1m, k2m])
    abs = tf.math.abs(subtracted)
    output = tf.keras.backend.sum(abs, axis=1)

    reverso = Dense(k * 5 * 3 * dims, activation=tf.nn.swish, name="Reverso")
    # reverso1 = Dense(1024, name="Reverso1")
    reverso_output = Dense(k * 5, name="ReversoOutput")
    reshaped = tf.keras.layers.Reshape((k, 5))

    reverso_layer = tf.keras.Sequential()
    reverso_layer.add(reverso)
    reverso_layer.add(reverso_output)
    reverso_layer.add(reshaped)

    # k1r = reshaped(reverso_output(reverso(k1m)))
    # k2r = reshaped(reverso_output(reverso(k2m)))

    k1r = reverso_layer(k1m)
    k2r = reverso_layer(k2m)

    model = Model(inputs=[model_input], outputs=[output, k1r, k2r])
    metrics = [
        [tf.keras.metrics.MeanAbsoluteError()],
        [tf.keras.metrics.CategoricalAccuracy()],
        [tf.keras.metrics.CategoricalAccuracy()],
    ]
    loss_weights = [1, 0.3, 0.3]
    model.compile(
        loss=[
            loss,
            tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        ],
        optimizer=opt,
        metrics=metrics,
        loss_weights=loss_weights,
    )
    return magic, reverso_layer, model


opt = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-2)
# opt = tf.keras.optimizers.experimental.Nadam()


magic, reverso_layer, model = model2(opt)
print(model.summary())
print("At first training step...")

print(opt.lr)

logcb = tf.keras.callbacks.CSVLogger(
    "log_k_{}_dims_{}_activation_{}_loss_{}_bias_.csv".format(
        k, dims, activation, loss, bias
    ),
    separator=",",
    append=True,
)


cur_epoch = 0
epochs = 256
model.fit(
    ds,
    initial_epoch=cur_epoch,
    validation_data=vds,
    epochs=cur_epoch + epochs,
    steps_per_epoch=2048,
    validation_steps=256,
    verbose=1,
    shuffle=False,
    callbacks=[logcb],
)

# weights = model.get_weights()
# for i in weights:
#    print(tf.shape(i))

# np.save(
#    "weights/wide_singlelayer_k{}_23Apr2023_linear_nucleotide_model_dims_{}_epochs_{}".format(
#        k, dims, epochs
#    ),
#    weights,
# )

magic_weights = magic.get_weights()

# for i in magic_weights:
#    print(tf.shape(i))

# print(magic_weights[0])
# print(tf.shape(magic_weights[0]))

np.save(
    "weights/wide_singlelayer_k{}_23Apr2023_{}_nucleotide_model_magic_dims_{}_epochs_{}_loss_{}_bias_{}".format(
        k, activation, dims, epochs, loss, bias
    ),
    magic_weights,
)

# magic.save_weights("weights/wide_singlelayer_k{}_23Apr2023_linear_nucleotide_model_magic_dims_{}_epochs_{}".format(k, dims, epochs))

reverso_weights = reverso_layer.get_weights()

filename = "weights/wide_singlelayer_k{}_23Apr2023_{}_nucleotide_model_reverso_dims_{}_epochs_{}_loss_{}_bias_{}".format(
    k, activation, dims, epochs, loss
)

reverso_layer.save_weights(filename)

# for i in reverso_weights:
#    print(tf.shape(i))

# np.save(
#    "weights/wide_singlelayer_k{}_23Apr2023_linear_nucleotide_model_reverso_dims_{}_epochs_{}".format(
#        k, dims, epochs
#    ),
#    reverso_weights,
# )
