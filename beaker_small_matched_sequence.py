# 12 Oct model
# Less layers!
# First layer can attend to self, the rest of the layers mask the self word...

import os
import aim

# os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

import tensorflow_addons as tfa
import numpy as np
import time
import pyracular

from lib.useful import calc_kmer_numeric_tuple, convert_tuple_to_string, calc_distance, convert_tuple_to_np, cos_sim, convert_string_to_nparray, convert_string_to_nparray_tuple
from lib.bert_inspired import get_angles, positional_encoding, fasta_generator, discriminator_layer, point_wise_feed_forward_network
from TransformerModelPosConcat_21Oct import Transformer, CustomSchedule
from tensorflow.keras.layers import Dense, Embedding, Flatten, Lambda, Subtract, Input, Concatenate, AveragePooling1D, LocallyConnected1D, Conv1D, GaussianNoise, BatchNormalization, Reshape, GlobalAveragePooling1D, Dropout
from tensorflow.keras.models import Model, Sequential

# Hyper parameters
k = 21
window_size = 32
num_layers = 4
embedding_dims = 32
output_dims = 128 # Output dims are also internal dims!
intermediate_dims = 256
num_heads = 4
dropout_rate = 0.15
max_positions = 512
batch_size = 32

#optimizer = tfa.optimizers.LAMB(learning_rate=0.001)
#optimizer = tf.keras.optimizers.Nadam()

#radam = tfa.optimizers.RectifiedAdam(learning_rate=0.0000001)
#ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
#optimizer = ranger

#lr = tf.keras.experimental.CosineDecay(4e-6, 256)
lr = tfa.optimizers.ExponentialCyclicalLearningRate(5e-8, 5e-6, 2048)

#optimizer = tfa.optimizers.LAMB(learning_rate=lr)
optimizer = tfa.optimizers.RectifiedAdam(learning_rate=lr)
optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=10)

transformer = Transformer(num_layers, embedding_dims, output_dims, num_heads, intermediate_dims, max_positions,
                          dropout=dropout_rate, attention_dropout=dropout_rate, activation=tfa.activations.mish)

magic = Dense(embedding_dims, 
                activation=tf.nn.swish, 
                name="Magic", 
                use_bias=False, 
                trainable=False,
                dtype=tf.float32)
EPOCHS = 12

batch_input = Input(shape=(2,window_size+1,k*5), dtype="float32", name="BatchInput")

contexts_a = magic(batch_input[:, 0])
contexts_b = magic(batch_input[:, 1])
enc_outputs_a, _, _ = transformer(contexts_a, True)
enc_outputs_b, _, _ = transformer(contexts_b, True)

def order_layer(neurons):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(neurons, activation=tfa.activations.mish),
      tf.keras.layers.Dense(1),
  ])

Order = order_layer(64)
drop = Dropout(dropout_rate)

out = Order(drop(tf.concat([enc_outputs_b[:, 0], enc_outputs_a[:, 0]], axis=-1)))

model = Model(inputs=[batch_input], outputs=[out])

#Load up the weights
weights = np.load("weights/weights_wide_singlelayer_k21_3Aug2020model_21_dims_32_epochs256.npy", allow_pickle=True)
magic.set_weights([weights[0][0]])

print(model.summary())

checkpoint_path = "beaker_small_nt_seqmatch" # /model_{epoch:04d}.ckpt"
latest = tf.train.latest_checkpoint(checkpoint_path)
if latest:
  print("Loading checkpoint")
  print(latest)
  model.load_weights(latest).expect_partial()
  print("Checkpoint loaded");

cls = np.asarray([[1] * 105])

def valid_gen():
  fasta = pyracular.MatchedKmersGenerator(k, "/mnt/ramfs/nt_validation.sfasta", window_size, 8192*8)
  for i in fasta:
    kmers = list()
    kmers.extend([np.concatenate([cls, i['kmers'][0]])])
    kmers.extend([np.concatenate([cls, i['kmers'][1]])])
    yield kmers, i['matched']
  print("=================Finished validation generator")

def gen():
  fasta = pyracular.MatchedKmersGenerator(k, "/mnt/ramfs/nt_train.sfasta", window_size, 8192*32)
  for i in fasta:
    kmers = list()
    kmers.extend([np.concatenate([cls, i['kmers'][0]])])
    kmers.extend([np.concatenate([cls, i['kmers'][1]])])
    yield kmers, i['matched']
  print("=================Finished generator=================")

x = valid_gen()
print(np.asarray(next(x)[0]).shape)

ds = tf.data.Dataset.from_generator(valid_gen, (tf.float32, tf.float32))
validation_generator = ds.repeat(8192*64).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))
training_generator = ds.repeat(8).shuffle(8192*16).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

checkpoint_path = "beaker_small_nt_seqmatch/model_{epoch:04d}.ckpt"

checkpointer = tf.keras.callbacks.ModelCheckpoint(
        save_freq=2048,
        filepath=checkpoint_path,
#        save_best_only=True,
        save_weights_only=True,
#        monitor="train_accuracy",
        verbose=0,
    )

class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        #logs.update({'lr': scheduler(epoch)})
        super().on_epoch_end(epoch, logs)

tensorboard_callback = CustomTensorBoard(
                    log_dir = "logs_seqmatch_beakersmall_23Oct",
                    update_freq=128,
                    write_images=True,
                    write_graph=False,
                    #profile_batch=(1200,1250)
                    )

csvlog = tf.keras.callbacks.CSVLogger(
   "BeakerSmallSeqMatch.tsv" , separator='\t', append=False
)

from aim.keras import AimTracker
session = aim.Session(experiment="BeakerSmall")
cb = AimTracker.metrics(session)

metrics = [tf.keras.metrics.BinaryCrossentropy(name='binary_crossentropy', from_logits=True, label_smoothing=0.3), "binary_accuracy"]#, tf.keras.metrics.TrueNegatives(), tf.keras.metrics.TruePositives()]
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#Warm-up!

lr = tfa.optimizers.ExponentialCyclicalLearningRate(1e-8, 1e-4, 1024)
optimizer = tfa.optimizers.LAMB(learning_rate=lr)

model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=metrics)

model.fit(x=training_generator,
          use_multiprocessing=True,
          shuffle=False,
          validation_data=validation_generator,
          validation_steps=1024,
          steps_per_epoch=2048,
          epochs=16,
          callbacks=[tensorboard_callback, checkpointer, csvlog, cb])

total_epochs = 16

# Run for awhile...
optimizer = tfa.optimizers.LAMB(learning_rate=1e-5, weight_decay_rate=0.01)

model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=metrics)

model.fit(x=training_generator,
          use_multiprocessing=True,
          shuffle=False,
          validation_data=validation_generator,
          validation_steps=1024,
          initial_epoch=total_epochs,
          steps_per_epoch=8192,
          epochs=1024+total_epochs,
          callbacks=[tensorboard_callback, checkpointer, csvlog, cb])

lr = tf.keras.experimental.CosineDecayRestarts(1e-4, 8192*2)

optimizer = tfa.optimizers.LAMB(learning_rate=lr, weight_decay_rate=0.001)
optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=10)

model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=metrics)

ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))
training_generator = ds.repeat(2).shuffle(8192*16).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

total_epochs = total_epochs + 1024

model.fit(x=training_generator,
          use_multiprocessing=True,
          shuffle=False,
          validation_data=validation_generator,
          initial_epoch=total_epochs,
          validation_steps=1024,
          steps_per_epoch=2048,
          epochs=total_epochs+1024,
          callbacks=[tensorboard_callback, checkpointer, csvlog, cb])
