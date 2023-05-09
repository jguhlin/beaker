print("Don't forget: set -x LD_LIBRARY_PATH $CONDA_PREFIX/lib/ ")

# 12 Oct model

import os

from biobeaker.utils import get_angles, positional_encoding
from biobeaker import BEAKER

import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras

import numpy as np
import time
import pyracular

# from lib.useful import (
#    calc_kmer_numeric_tuple,
#    convert_tuple_to_string,
#    calc_distance,
#    convert_tuple_to_np,
#    cos_sim,
#    convert_string_to_nparray,
#    convert_string_to_nparray_tuple,
# )
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
intermediate_dims = 1024
num_heads = 8
dropout_rate = 0.15
max_positions = 512
batch_size = 32

transformer = BEAKER(
    num_layers,
    embedding_dims,
    output_dims,
    num_heads,
    intermediate_dims,
    max_positions,
    dropout=dropout_rate,
    attention_dropout=dropout_rate,
    activation=tf.keras.activations.gelu,
)

generator = BEAKER(
    6,
    embedding_dims,
    output_dims,
    8,
    768,
    max_positions,
    dropout=0.15,
    attention_dropout=0.15,
    activation=tf.keras.activations.gelu,
)


def matched_layer():
    return tf.keras.Sequential(
        [
            tfp.layers.DenseFlipout(1024),
            tfp.layers.DenseFlipout(1024, activation="swish"),
            tfp.layers.DenseFlipout(1, activation="linear"),
        ],
        name="Matched",
    )


def rc_layer(neurons, activation="swish"):
    return tf.keras.Sequential(
        [
            tfp.layers.DenseFlipout(neurons),
            tfp.layers.DenseFlipout(neurons, activation=activation),
            tfp.layers.DenseFlipout(1, activation="linear"),
        ],
        name="Rc",
    )


def discriminator_layer(neurons, activation="swish"):
    return tf.keras.Sequential(
        [
            tfp.layers.DenseFlipout(neurons),
            tfp.layers.DenseFlipout(neurons, activation=activation),
            tfp.layers.DenseFlipout(1, activation="linear"),
        ],
        name="Discriminator",
    )


# def reverso_layer():
#    return tf.keras.Sequential(
#        [
#            tf.keras.layers.Dense(
#                k * 5 * 3 * embedding_dims, activation=tf.nn.swish, name="Reverso"
#            ),
#            tf.keras.layers.Dense(k * 5, name="ReversoOutput"),
#            #tf.keras.layers.Reshape((window_size, k, 5)),
#            #tf.keras.layers.Softmax(axis=-1),
#            tf.keras.layers.Reshape((window_size, k, 5)),
#        ],
#        name="Reverso",
#    )

reverso = Dense(k * 5 * 3 * embedding_dims, activation=tf.nn.swish, name="Reverso")
# reverso1 = Dense(1024, name="Reverso1")
reverso_output = Dense(k * 5, name="ReversoOutput")
reshaped = tf.keras.layers.Reshape((window_size, k, 5))
reshaped.trainable = False
reverso_output.trainable = False
reverso.trainable = False

reverso_layer = tf.keras.Sequential(name="Reverso")
reverso_layer.add(reverso)
reverso_layer.add(reverso_output)
reverso_layer.add(reshaped)

reverso.trainable = False


magic = Dense(
    embedding_dims,
    activation="linear",
    name="Magic",
    use_bias=False,
    trainable=False,
    dtype=tf.float32,
)
EPOCHS = 12

# cls = np.asarray([[1] * 105])
init = tfk.initializers.RandomNormal()
cls = tf.Variable(
    trainable=True,
    name="CLS Token",
    shape=(1, embedding_dims),
    initial_value=init(shape=(1, embedding_dims)),
)

# Define the model
# Input is 2 with mask (and CLS token)
# input is another 2 without mask (also with CLS token)
batch_input = Input(shape=(2, window_size, k * 5), dtype="float32", name="BatchInput")

seq_mask = Input(shape=(2, window_size), dtype="float32", name="SeqMask")
truth = Input(shape=(2, window_size, k * 5), dtype="float32", name="BatchInputTrue")
masked_tokens = Input(
    shape=(2, window_size, k * 5), dtype="float32", name="GeneratorMaskedTokens"
)

contexts_a = magic(batch_input[:, 0])
contexts_b = magic(batch_input[:, 1])

contexts_a_true = magic(truth[:, 0])
contexts_b_true = magic(truth[:, 1])

BackToEmbeddings = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(1024),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.Dense(embedding_dims, use_bias=False)
    ])


# Generator - Train to replace masked token
generated_a, _, _ = generator(contexts_a, training=True, mask=seq_mask[:, 0])
generated_b, _, _ = generator(contexts_b, training=True, mask=seq_mask[:, 1])
generated_a = BackToEmbeddings(generated_a)
generated_b = BackToEmbeddings(generated_b)

# Need to softmax generated_a, b to one-hot embeddings to feed into the discriminator
# Also probably a good idea to train on the embedding weights rather than the one-hot kmers, need to think about it
# Also need to replace the [MASK] tokens with what the generator guesses, so that it's not entirely random (prevent generator and discrimiinator talking to each other)

# No longer using CLS Token here, so no need to drop it
generator1_reversed = reverso_layer(generated_a)  # * mask[:, 0]
generator2_reversed = reverso_layer(generated_b)  # * mask[:, 1]

indices_a = tf.where(masked_tokens[:, 0] == 0)
indices_b = tf.where(masked_tokens[:, 1] == 0)

updates_a = tfk.layers.Flatten()(tf.gather_nd(generator1_reversed, indices_a))
updates_b = tfk.layers.Flatten()(tf.gather_nd(generator2_reversed, indices_b))

modified_a = magic(
    tf.tensor_scatter_nd_update(truth[:, 0], indices_a, tf.squeeze(updates_a))
)
modified_b = magic(
    tf.tensor_scatter_nd_update(truth[:, 1], indices_b, tf.squeeze(updates_b))
)

# Prepend CLS token
cls_b = tf.broadcast_to(cls, (batch_size, 1, embedding_dims))
modified_a_cls = tf.concat([cls_b, modified_a], axis=-2)
modified_b_cls = tf.concat([cls_b, modified_b], axis=-2)

print(tf.shape(seq_mask))

# Need to make sure the cls token is not masked out
cls_mask = tf.broadcast_to(1.0, (batch_size, 1))
print(tf.shape(cls_mask))
seq_mask_a = tf.concat([cls_mask, seq_mask[:, 0]], axis=-1)
seq_mask_b = tf.concat([cls_mask, seq_mask[:, 1]], axis=-1)
print(tf.shape(seq_mask_a))

enc_outputs_a, _, _ = transformer(modified_a_cls, training=True, mask=seq_mask_a)
enc_outputs_b, _, _ = transformer(modified_b_cls, training=True, mask=seq_mask_b)

Matched = matched_layer()

Rc = rc_layer(1024)
DropRc = Dropout(dropout_rate)

conv1d = tfp.layers.Convolution1DFlipout(512, 5, padding="same")

Discriminator = discriminator_layer(512)
DropDiscriminator = Dropout(dropout_rate)

# CosSim = tf.keras.layers.Dot(axes=-1, normalize=True)
# CosSimNoise = tf.keras.layers.GaussianNoise(0.05)

# out1 = Matched(DropMatched(tf.concat([enc_outputs_b[:, 0], enc_outputs_a[:, 0]], axis=-1)))
# out1 = CosSim([enc_outputs_a[:, 0], enc_outputs_b[:, 0]])
out1 = Matched(tf.concat([enc_outputs_a[:, 0], enc_outputs_b[:, 0]], axis=-1))
out2 = Rc(tf.concat([enc_outputs_b[:, 0], enc_outputs_a[:, 0]], axis=-1))
# out1 = Matched(out1)

out0a = tf.squeeze(Discriminator(conv1d(enc_outputs_a[:, 1:])), name="Dis0")
out0b = tf.squeeze(Discriminator(conv1d(enc_outputs_b[:, 1:])), name="Dis1")

Reverso = reverso_layer()

generator1_reversed = Reverso(generated_a[:, 1:])  # * mask[:, 0]
generator2_reversed = Reverso(generated_b[:, 1:])  # * mask[:, 1]

# gen_loss_a = tf.math.reduce_sum(tf.math.square(generated_a - contexts_a_true), axis=-1)
# gen_loss_b = tf.math.reduce_sum(tf.math.square(generated_b - contexts_b_true), axis=-1)

model = Model(
    inputs=[batch_input, seq_mask, truth, masked_tokens],
    outputs=[
        out0a,
        out0b,
        out1,
        out2,
        generator1_reversed,
        generator2_reversed,
    ],  # , generated_a, generated_b],
)

# Load up the weights
weights = np.load(
    # "weights/weights_wide_singlelayer_k21_3Aug2020model_21_dims_32_epochs256.npy",
    "weights/wide_singlelayer_k21_23Apr2023_linear_nucleotide_model_magic_dims_32_epochs_256.npy",
    allow_pickle=True,
)
magic.set_weights([weights[0]])

reverso_layer.load_weights(
    "weights/wide_singlelayer_k21_23Apr2023_linear_nucleotide_model_reverso_dims_32_epochs_256"
)

# Define the generators
# cls = np.asarray([[1] * 105])



def valid_gen():
    fasta = pyracular.TripleLossKmersGenerator(
        k,
        "../nt.sfasta",
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
        # kmers.extend([np.concatenate([cls, i.kmers1]).tolist()])
        # kmers.extend([np.concatenate([cls, i.kmers2]).tolist()])
        # kmers.extend([np.concatenate([cls, i.kmers3]).tolist()])
        # kmers.extend([np.concatenate([cls, i.kmers4]).tolist()])

        yield kmers, (i.truth1, i.truth2, i.matched, i.reversecomplement, 0, 0)
    print("=================Finished Training generator=================")


fakemask = np.ones((2, window_size))

# TODO: Currently loss needs to account for using masks
# i.e., need to broadcast and multiply by 0 for parts of the sequences
def gen():
    fasta = pyracular.TripleLossKmersGenerator(
        k,
        "../nt.sfasta",
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

        # kmers = list()
        # kmers.extend([np.concatenate([cls, i.kmers1]).tolist()])
        # kmers.extend([np.concatenate([cls, i.kmers2]).tolist()])
        kmers = [i.kmers1, i.kmers2]

        # kmers_true = list()
        # kmers_true.extend([np.concatenate([cls, i.kmers3]).tolist()])
        # kmers_true.extend([np.concatenate([cls, i.kmers4]).tolist()])
        kmers_true = [i.kmers3, i.kmers4]

        yield (kmers, fakemask, kmers_true, [i.truth1, i.truth2]), (
            i.truth1,
            i.truth2,
            i.matched,
            i.reversecomplement,
            np.reshape(i.kmers3, (window_size, k, 5)),
            np.reshape(i.kmers4, (window_size, k, 5)),
            #            np.dot(kmers_true[0], weights[0][0]),
            #            np.dot(kmers_true[1], weights[0][0])
        )
    print("=================Finished Training generator=================")


output_sig = (
    (
        tf.TensorSpec(shape=(2, window_size, k * 5), dtype=tf.int16),
        tf.TensorSpec(shape=(2, window_size), dtype=tf.int16),
        tf.TensorSpec(shape=(2, window_size, k * 5), dtype=tf.int16),
        tf.TensorSpec(shape=(2, window_size), dtype=tf.int16),
        # tf.TensorSpec(shape=window_size, dtype=tf.int16),
        # tf.TensorSpec(shape=window_size, dtype=tf.int16),
    ),
    (
        tf.TensorSpec(shape=window_size, dtype=tf.int16),
        tf.TensorSpec(shape=window_size, dtype=tf.int16),
        tf.TensorSpec(shape=(), dtype=tf.int16),
        tf.TensorSpec(shape=(), dtype=tf.int16),
        tf.TensorSpec(shape=(window_size, k, 5), dtype=tf.float32),
        tf.TensorSpec(shape=(window_size, k, 5), dtype=tf.float32),
        #        tf.TensorSpec(shape=(window_size+1, embedding_dims), dtype=tf.float32),
        #        tf.TensorSpec(shape=(window_size+1, embedding_dims), dtype=tf.float32),
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
# optimizer = tfa.optimizers.LAMB(learning_rate=lr)

# Copied from
# https://github.com/tensorflow/models/blob/v2.12.0/official/modeling/optimization/lr_schedule.py#L92-L162
# as tensorflow-models-official is not installing from pip as of 9 May 2023
from typing import Mapping, Any, Union, Optional
class LinearWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Linear warmup schedule."""

  def __init__(self,
               after_warmup_lr_sched: Union[
                   tf.keras.optimizers.schedules.LearningRateSchedule, float],
               warmup_steps: int,
               warmup_learning_rate: float,
               name: Optional[str] = None):
    super().__init__()
    self._name = name
    self._after_warmup_lr_sched = after_warmup_lr_sched
    self._warmup_steps = warmup_steps
    self._init_warmup_lr = warmup_learning_rate
    if isinstance(after_warmup_lr_sched,
                  tf.keras.optimizers.schedules.LearningRateSchedule):
      self._final_warmup_lr = after_warmup_lr_sched(warmup_steps)
    else:
      self._final_warmup_lr = tf.cast(after_warmup_lr_sched, dtype=tf.float32)

  def __call__(self, step: int):

    global_step = tf.cast(step, dtype=tf.float32)

    linear_warmup_lr = (
        self._init_warmup_lr + global_step / self._warmup_steps *
        (self._final_warmup_lr - self._init_warmup_lr))

    if isinstance(self._after_warmup_lr_sched,
                  tf.keras.optimizers.schedules.LearningRateSchedule):
      after_warmup_lr = self._after_warmup_lr_sched(step)
    else:
      after_warmup_lr = tf.cast(self._after_warmup_lr_sched, dtype=tf.float32)

    lr = tf.cond(global_step < self._warmup_steps,
                 lambda: linear_warmup_lr,
                 lambda: after_warmup_lr)
    return lr

  def get_config(self) -> Mapping[str, Any]:
    if isinstance(self._after_warmup_lr_sched,
                  tf.keras.optimizers.schedules.LearningRateSchedule):
      config = {
          "after_warmup_lr_sched": self._after_warmup_lr_sched.get_config()}  # pytype: disable=attribute-error
    else:
      config = {"after_warmup_lr_sched": self._after_warmup_lr_sched}  # pytype: disable=attribute-error

    config.update({
        "warmup_steps": self._warmup_steps,
        "warmup_learning_rate": self._init_warmup_lr,
        "name": self._name
    })
    return config


csd = tf.keras.optimizers.schedules.CosineDecayRestarts(5e-5, 8192, alpha=1e-8)
learning_rate = LinearWarmup(csd, 20000, 5e-5)

# Seems to work
optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)

# Seems to fail
# optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)

loss0a = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # Dis0
loss0b = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # Dis1
loss1 = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # Matched
loss2 = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # RC
loss3a = tf.keras.losses.CategoricalCrossentropy(from_logits=True)  # Gen1
loss3b = tf.keras.losses.CategoricalCrossentropy(from_logits=True)  # Gen2
# loss4a = tf.keras.losses.MeanSquaredError()
# loss4b = tf.keras.losses.MeanSquaredError()

loss = [loss0a, loss0b, loss1, loss2, loss3a, loss3b]  # , loss4a, loss4b]
# loss_weights=[0.75,0.75,1,1]
loss_weights = [0.5, 0.5, 1.0, 1.0, 0.5, 0.5]

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
        tf.keras.metrics.CategoricalAccuracy(),
        tf.keras.metrics.TrueNegatives(),
        tf.keras.metrics.TruePositives(),
        tf.keras.metrics.FalseNegatives(),
        tf.keras.metrics.FalsePositives(),
    ],
    [
        tf.keras.metrics.CategoricalAccuracy(),
        tf.keras.metrics.TrueNegatives(),
        tf.keras.metrics.TruePositives(),
        tf.keras.metrics.FalseNegatives(),
        tf.keras.metrics.FalsePositives(),
    ],  # [], []
]

model.compile(
    loss=loss,
    loss_weights=loss_weights,
    optimizer=optimizer,
    metrics=metrics,
)

print(model.summary())

model.fit(
    x=training_generator,
    use_multiprocessing=True,
    shuffle=False,
    batch_size=256,
    validation_steps=2048,
    steps_per_epoch=4096,
    epochs=8192 * 2,
    callbacks=[
        # tensorboard_callback,
        checkpointer,
        # csvlog,
    ],
)  # cb

total_epochs = 1024

# Run for awhile...
lr = tf.keras.experimental.CosineDecayRestarts(1e-4, 8192 * 2)
# optimizer = tfa.optimizers.LAMB(learning_rate=lr, weight_decay_rate=0.01)
optimizer = tf.keras.optimizers.AdamW(1e-4, 1e-6)

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

# optimizer = tfa.optimizers.LAMB(learning_rate=lr, weight_decay_rate=0.001)
# optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=10)
optimizer = tf.keras.optimizers.AdamW(1e-4, 1e-6)

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
