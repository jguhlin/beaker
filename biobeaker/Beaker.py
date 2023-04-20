import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Lambda,
    Subtract,
    Input,
    Concatenate,
    BatchNormalization,
    Reshape,
)
from .utils import (
    get_angles,
    positional_encoding,
    discriminator_layer,
)


def ffn(output_dims, intermediate_dims, activation=tf.nn.swish):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(intermediate_dims, activation=activation),
            tf.keras.layers.Dense(output_dims),
        ]
    )


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        output_dims,
        intermediate_dims,
        num_heads,
        dropout,
        attention_dropout,
        activation,
    ):
        super(EncoderLayer, self).__init__()
        self.supports_masking = True

        self.mha = tfa.layers.MultiHeadAttention(
            head_size=intermediate_dims,
            num_heads=num_heads,
            output_size=intermediate_dims,
            dropout=attention_dropout,
            return_attn_coef=True,
            dtype=tf.float32,
        )

        self.ffn = ffn(output_dims, intermediate_dims, activation)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False, mask=None):
        if mask is None:
            broadcast_float_mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
            x = x * broadcast_float_mask

        attn, attn_weights = self.mha([x, x], mask=mask, training=training)
        out1 = self.layernorm1(x + attn, training=training)

        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output, training=training)

        if mask is not None:
            out2 = out2 * broadcast_float_mask

        return out2, attn_weights


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        embedding_dims,
        output_dims,
        num_heads,
        intermediate_dims,
        maximum_positions=256,
        dropout=0.1,
        attention_dropout=0.1,
        positional_encoding_dims=16,
        activation=tf.nn.swish,
    ):
        super(Encoder, self).__init__()

        self.supports_masking = True
        self.embedding_dims = embedding_dims
        self.num_layers = num_layers
        self.output_dims = output_dims
        self.intermediate_dims = intermediate_dims
        self.final_dense = Dense(output_dims)

        self.pos_encoding = positional_encoding(
            maximum_positions, positional_encoding_dims
        )

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dense1 = Dense(intermediate_dims - positional_encoding_dims)

        self.enc_layers = [
            EncoderLayer(
                output_dims,
                intermediate_dims,
                num_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation=activation,
            )
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False, mask=None):
        self.pos_encoding = tf.cast(self.pos_encoding, x.dtype)

        seq_len = tf.shape(x)[1]

        x = self.dense1(x, training=training)
        x *= tf.math.sqrt(tf.cast(self.intermediate_dims, x.dtype))
        x = self.dropout(x, training=training)

        # Instead of adding the positional encodings, we just concat them the the embeddings
        # It's worked better in my earlier trials, so I switched early on to this method.
        y = tf.squeeze(self.pos_encoding[:, :seq_len, :], 0)
        x = tf.map_fn(lambda z: tf.concat([z, y], 1), x)

        attention_weights = {}
        encoder_outputs = []

        for i in range(self.num_layers):
            x, block = self.enc_layers[i](x, training=training, mask=mask)
            encoder_outputs.append(x)
            attention_weights["layer_{}_attention".format(i + 1)] = block
        x = self.final_dense(x)

        return x, attention_weights, encoder_outputs


class BEAKER(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        embedding_dims,
        output_dims,
        num_heads,
        intermediate_dims,
        max_positions,
        dropout=0.1,
        attention_dropout=0.1,
        positional_encoding_dims=16,
        activation=tf.nn.swish,
    ):
        super(BEAKER, self).__init__()

        self.encoder = Encoder(
            num_layers,
            embedding_dims,
            output_dims,
            num_heads,
            intermediate_dims,
            max_positions,
            dropout,
            attention_dropout,
            positional_encoding_dims,
            activation,
        )

    def call(self, inp, training=False, mask=None):
        enc_output, attention_weights, all_outputs = self.encoder(inp, training, mask)
        if mask is not None:
            broadcast_float_mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
            enc_output = enc_output * broadcast_float_mask
        return enc_output, attention_weights, all_outputs
