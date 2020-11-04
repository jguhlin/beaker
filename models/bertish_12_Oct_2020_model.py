# 12 Oct model
# Less layers!
# First layer can attend to self, the rest of the layers mask the self word...

import os

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
from TransformerModelPosConcat_12Oct import Transformer, CustomSchedule
from tensorflow.keras.layers import Dense, Embedding, Flatten, Lambda, Subtract, Input, Concatenate, AveragePooling1D, LocallyConnected1D, Conv1D, GaussianNoise, BatchNormalization, Reshape
from tensorflow.keras.models import Model, Sequential

# Hyper parameters
k = 21
window_size = 64
replacement_pct = 0.15
num_layers = 6
embedding_dims = 32
output_dims = 128 # Output dims are also internal dims!
intermediate_dims = 256
num_heads = 12
dropout_rate = 0.10
max_positions = 256
batch_size = 128

scheduler = CustomSchedule(intermediate_dims, warmup_steps=10000)
optimizer = tfa.optimizers.LAMB(learning_rate=0.001)

cosine_decay_restarts = tf.keras.experimental.CosineDecayRestarts(
    0.005, 8192 * 10, t_mul=2.0, m_mul=1.0, alpha=0.0,
    name=None)

#radam = tfa.optimizers.RectifiedAdam(learning_rate=0.0005)
#ranger = tfa.optimizers.Lookahead(radam, sync_period=32, slow_step_size=0.5)
#optimizer = ranger

#optimizer = tf.keras.optimizers.Nadam()

discriminator = discriminator_layer(4096, window_size, output_dims)

transformer = Transformer(num_layers, embedding_dims, output_dims, num_heads, intermediate_dims, max_positions,
                          dropout=dropout_rate, attention_dropout=dropout_rate, activation=tfa.activations.mish)

magic = Dense(embedding_dims, 
                activation=tf.nn.swish, 
                name="Magic", 
                use_bias=False, 
                trainable=False,
                dtype=tf.float32)
EPOCHS = 12

batch_input = Input(shape=(window_size,k*5), dtype="float32", name="BatchInput")

contexts = magic(batch_input)
enc_outputs, _, all_enc_outputs = transformer(contexts, True)
discriminator_output = discriminator(enc_outputs) #tf.reshape(enc_outputs, (batch_size, output_dims*window_size)))
discriminator_output = tf.squeeze(discriminator_output)
#predictions = Reshape((window_size, 1), name="Reshape_Predictions")(discriminator_output)

model = Model(inputs=[batch_input], outputs=[discriminator_output])
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=optimizer,
    metrics=["binary_crossentropy", "binary_accuracy"])

#Load up the weights
weights = np.load("weights/weights_wide_singlelayer_k21_3Aug2020model_21_dims_32_epochs256.npy", allow_pickle=True)
magic.set_weights([weights[0][0]])

print(model.summary())

checkpoint_path = "bertish_12_Oct_2020_nt" # /model_{epoch:04d}.ckpt"
latest = tf.train.latest_checkpoint(checkpoint_path)
if latest:
  print("Loading checkpoint")
  print(latest)
  model.load_weights(latest).expect_partial()
  print("Checkpoint loaded");

def valid_gen():
  fasta = pyracular.MaskedKmersGenerator(k, "/mnt/ramfs/nt_validation.sfasta", window_size, replacement_pct, True, 8192*8)
  for i in fasta:
    yield i['kmers'], i['truth']
  print("=================Finished validation generator")

def gen():
  fasta = pyracular.MaskedKmersGenerator(k, "/mnt/ramfs/nt_train.sfasta", window_size, replacement_pct, True, 8192*32)
  for i in fasta:
    yield i['kmers'], i['truth']
  print("=================Finished generator=================")

ds = tf.data.Dataset.from_generator(valid_gen, (tf.float32, tf.float32))
validation_generator = ds.batch(4096).prefetch(tf.data.experimental.AUTOTUNE)

ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))
training_generator = ds.repeat(16).shuffle(8192*16).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

checkpoint_path = "bertish_12_Oct_2020_nt/model_{epoch:04d}.ckpt"

checkpointer = tf.keras.callbacks.ModelCheckpoint(
        save_freq=2048,
        filepath=checkpoint_path,
#        save_best_only=True,
        save_weights_only=True,
#        monitor="train_accuracy",
        verbose=0,
    )


eds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32)).batch(2048)
eds = iter(eds)
ex = next(eds)
print(ex)

ys = model.predict(ex[0])

print(ys)
print(ex[1])


class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        #logs.update({'lr': scheduler(epoch)})
        super().on_epoch_end(epoch, logs)

tensorboard_callback = CustomTensorBoard(
                    log_dir = "logs_12Oct",
                    update_freq=128,
                    write_images=True,
                    write_graph=False,
                    #profile_batch=(1200,1250)
                    )

model.fit(x=training_generator,
          use_multiprocessing=True,
          shuffle=False,
          validation_data=validation_generator,
          validation_steps=16,
          steps_per_epoch=4096,
          epochs=8192,
          callbacks=[tensorboard_callback, checkpointer])


