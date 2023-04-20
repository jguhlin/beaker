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

def reverso_layer():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                k * 5 * 3 * 32, use_bias=False, activation=tf.nn.swish, name="Reverso0"
            ),
            tf.keras.layers.Dense(k * 5 * 3 * 8, use_bias=False, activation="relu", name="Reverso1"),
            tf.keras.layers.Dense(k * 5, name="ReversoOutput"),
            tf.keras.layers.Reshape((window_size, k, 5)),
            tf.keras.layers.Softmax(axis=-1),
            tf.keras.layers.Reshape((window_size, k*5))
        ],
        name="Reverso"
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
# Input is 2 with mask (and CLS token)
# input is another 2 without mask (also with CLS token)
batch_input = Input(
    shape=(2, window_size + 1, k * 5), dtype="float32", name="BatchInput"
)

mask = Input(
    shape=(2, window_size), dtype="float32", name="Mask"
)

contexts_a = magic(batch_input[:, 0])
contexts_b = magic(batch_input[:, 1])

BackToEmbeddings = tf.keras.layers.Dense(embedding_dims, use_bias=False)

# Generator - Train to replace mask token
generated_a, _, _ = generator(contexts_a, training=True)
generated_b, _, _ = generator(contexts_b, training=True)
generated_a = BackToEmbeddings(generated_a)
generated_b = BackToEmbeddings(generated_b)

enc_outputs_a, _, _ = transformer(generated_a, training=True)
enc_outputs_b, _, _ = transformer(generated_b, training=True)

Matched = matched_layer()

Rc = rc_layer(1024)
DropRc = Dropout(dropout_rate)

Discriminator = discriminator_layer(1024)
DropDiscriminator = Dropout(dropout_rate)

CosSim = tf.keras.layers.Dot(axes=-1, normalize=True)
CosSimNoise = tf.keras.layers.GaussianNoise(0.05)

# TODO: I think this is only looking at the CLS token...
# out1 = Matched(DropMatched(tf.concat([enc_outputs_b[:, 0], enc_outputs_a[:, 0]], axis=-1)))
out1 = Matched(CosSimNoise(CosSim([enc_outputs_a[:, 0], enc_outputs_b[:, 0]])))
out2 = Rc(DropRc(tf.concat([out1, enc_outputs_b[:, 0], enc_outputs_a[:, 0]], axis=-1)))

out0a = tf.squeeze(Discriminator(DropDiscriminator(enc_outputs_a[:, 1:])), name="Dis0")
out0b = tf.squeeze(Discriminator(DropDiscriminator(enc_outputs_b[:, 1:])), name="Dis1")

Reverso = reverso_layer()

generator1_reversed = Reverso(generated_a[:, 1:]) #* mask[:, 0]
generator2_reversed = Reverso(generated_b[:, 1:]) #* mask[:, 1]

#gen_loss_a = tf.math.reduce_sum(tf.math.square(generated_a - contexts_a_true), axis=-1)
#gen_loss_b = tf.math.reduce_sum(tf.math.square(generated_b - contexts_b_true), axis=-1)

model = Model(
    inputs=[batch_input, mask], 
    outputs=[out0a, out0b, out1, out2, generator1_reversed, generator2_reversed]
)

# Load up the weights
weights = np.load(
    "weights/weights_wide_singlelayer_k21_3Aug2020model_21_dims_32_epochs256.npy",
    allow_pickle=True,
)
magic.set_weights([weights[0][0]])

# Define the generators
cls = np.asarray([[1] * 105])

def valid_gen():
    fasta = pyracular.TripleLossKmersGenerator(
        k,
        "/Volumes/archive/deardenlab/guhlin/nt.sfasta",
        0.15,
        window_size,
        8192,
        2,
        42,
    )
    for i in fasta:
        if len(i.kmers1) != window_size or len(i.kmers2) != window_size:
            continue
        kmers = list()
        kmers.extend([np.concatenate([cls, i.kmers1]).tolist()])
        kmers.extend([np.concatenate([cls, i.kmers2]).tolist()])
        kmers.extend([np.concatenate([cls, i.kmers3]).tolist()])
        kmers.extend([np.concatenate([cls, i.kmers4]).tolist()])

        yield kmers, (i.truth1, i.truth2, i.matched, i.reversecomplement, 0, 0)
    print("=================Finished Training generator=================")

fakemask = np.ones((2, window_size))

def gen():
    fasta = pyracular.TripleLossKmersGenerator(
        k,
        "/Volumes/archive/deardenlab/guhlin/nt.sfasta",
        0.15,
        window_size,
        8192,
        4,
        42,
    )
    for i in fasta:
        # print(len(i.kmers1), len(i.kmers2), len(i.truth1), len(i.truth2))
        # Until we do masking
        if len(i.kmers1) != window_size or len(i.kmers2) != window_size:
            continue
        if len(i.truth1) != window_size or len(i.truth2) != window_size:
            continue

        kmers = list()
        kmers.extend([np.concatenate([cls, i.kmers1]).tolist()])
        kmers.extend([np.concatenate([cls, i.kmers2]).tolist()])
        # kmers.extend([np.concatenate([cls, i.kmers3]).tolist()])
        # kmers.extend([np.concatenate([cls, i.kmers4]).tolist()])

        yield (kmers, fakemask), (i.truth1, i.truth2, i.matched, i.reversecomplement, i.kmers3, i.kmers4)
    print("=================Finished Training generator=================")


output_sig = (
    (tf.TensorSpec(shape=(2, window_size + 1, k * 5), dtype=tf.int16),
        tf.TensorSpec(shape=(2, window_size), dtype=tf.int16),
        # tf.TensorSpec(shape=window_size, dtype=tf.int16),
    ),
    (
        tf.TensorSpec(shape=window_size, dtype=tf.int16),
        tf.TensorSpec(shape=window_size, dtype=tf.int16),
        tf.TensorSpec(shape=(), dtype=tf.int16),
        tf.TensorSpec(shape=(), dtype=tf.int16),
        tf.TensorSpec(shape=(window_size, k * 5), dtype=tf.float32),
        tf.TensorSpec(shape=(window_size, k * 5), dtype=tf.float32),
    ),
)

# ds = tf.data.Dataset.from_generator(valid_gen, output_signature=output_sig)
# validation_generator = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

ds = tf.data.Dataset.from_generator(gen, output_signature=output_sig)
training_generator = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

checkpoint_path = "beaker_medium_nt_triple2023_generator/model_{epoch:04d}.ckpt"

latest = tf.train.latest_checkpoint("beaker_medium_nt_triple2023_generator/")
if latest:
    print("Loading checkpoint")
    print(latest)
    model.load_weights(latest).expect_partial()
    print("Checkpoint loaded")
else:
    print("Checkpoint NOT loaded")

checkpointer = tf.keras.callbacks.ModelCheckpoint(
    save_freq=4096,
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=0,
)


class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # logs.update({'lr': scheduler(epoch)})
        super().on_epoch_end(epoch, logs)


tensorboard_callback = CustomTensorBoard(
    log_dir="BeakerMediumTripleGenerator_tb",
    update_freq=128,
    write_images=True,
    write_graph=True,
)

csvlog = tf.keras.callbacks.CSVLogger(
    "BeakerMediumTripleGenerator.tsv", separator="\t", append=False
)

# from aim.keras import AimTracker
# session = aim.Session(experiment="BeakerMediumTriple_ws32")
# cb = AimTracker.metrics(session)

# lr = tfa.optimizers.ExponentialCyclicalLearningRate(1e-8, 1e-4, 2048)
lr = tf.keras.experimental.CosineDecayRestarts(1e-4, 8192 * 3)
optimizer = tfa.optimizers.LAMB(learning_rate=lr)

loss0a = tf.keras.losses.BinaryCrossentropy(from_logits=False)  # Dis0
loss0b = tf.keras.losses.BinaryCrossentropy(from_logits=False)  # Dis1
loss1 = tf.keras.losses.BinaryCrossentropy(from_logits=False)  # Matched
loss2 = tf.keras.losses.BinaryCrossentropy(from_logits=False)  # RC
loss3a = tf.keras.losses.BinaryCrossentropy(from_logits=False)  # Gen1
loss3b = tf.keras.losses.BinaryCrossentropy(from_logits=False)  # Gen2


loss = [loss0a, loss0b, loss1, loss2, loss3a, loss3b]
# loss_weights=[0.75,0.75,1,1]

metrics = [
    [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.TrueNegatives(),
        tf.keras.metrics.TruePositives(),
        tf.keras.metrics.FalseNegatives(),
        tf.keras.metrics.FalsePositives(),
    ],
    [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.TrueNegatives(),
        tf.keras.metrics.TruePositives(),
        tf.keras.metrics.FalseNegatives(),
        tf.keras.metrics.FalsePositives(),
    ],
    [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.TrueNegatives(),
        tf.keras.metrics.TruePositives(),
        tf.keras.metrics.FalseNegatives(),
        tf.keras.metrics.FalsePositives(),
    ],
    [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.TrueNegatives(),
        tf.keras.metrics.TruePositives(),
        tf.keras.metrics.FalseNegatives(),
        tf.keras.metrics.FalsePositives(),
    ],
    [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.TrueNegatives(),
        tf.keras.metrics.TruePositives(),
        tf.keras.metrics.FalseNegatives(),
        tf.keras.metrics.FalsePositives(),
    ],
    [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.TrueNegatives(),
        tf.keras.metrics.TruePositives(),
        tf.keras.metrics.FalseNegatives(),
        tf.keras.metrics.FalsePositives(),
    ]
]

model.compile(
    loss=loss,
    #    loss_weights=loss_weights,
    optimizer=optimizer,
    metrics=metrics,
)

model.fit(
    x=training_generator,
    use_multiprocessing=True,
    shuffle=False,
    batch_size=256,
    validation_steps=2048,
    steps_per_epoch=4096,
    epochs=8192 * 2,
    callbacks=[
        tensorboard_callback,
        checkpointer,
        csvlog,
    ],
)  # cb

total_epochs = 1024

# Run for awhile...
lr = tf.keras.experimental.CosineDecayRestarts(1e-4, 8192 * 2)
optimizer = tfa.optimizers.LAMB(learning_rate=lr, weight_decay_rate=0.01)

model.compile(
    loss=loss,
    #    loss_weights=loss_weights,
    optimizer=optimizer,
    metrics=metrics,
)

model.fit(
    x=training_generator,
    use_multiprocessing=True,
    shuffle=False,
    validation_data=validation_generator,
    initial_epoch=total_epochs,
    epochs=1024 + total_epochs,
    validation_steps=1024,
    steps_per_epoch=2048,
    callbacks=[tensorboard_callback, checkpointer, csvlog, cb],
)  # cb

lr = tf.keras.experimental.CosineDecayRestarts(1e-4, 8192 * 2)

optimizer = tfa.optimizers.LAMB(learning_rate=lr, weight_decay_rate=0.001)
optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=10)

model.compile(
    loss=loss, loss_weights=loss_weights, optimizer=optimizer, metrics=metrics
)

total_epochs = total_epochs + 1024

model.fit(
    x=training_generator,
    use_multiprocessing=True,
    shuffle=False,
    validation_data=validation_generator,
    initial_epoch=total_epochs,
    validation_steps=1024,
    steps_per_epoch=2048,
    epochs=total_epochs + 1024,
    callbacks=[tensorboard_callback, checkpointer, csvlog, cb],
)
