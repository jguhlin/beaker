import tensorflow as tf
import numpy as np
import pytest

import biobeaker
from biobeaker.utils import (
    get_angles,
    positional_encoding,
    calc_kmer_numeric_tuple,
    convert_tuple_to_string,
    convert_tuple_to_np,
    convert_string_to_nparray,
    convert_string_to_nparray_tuple,
)
from biobeaker import BEAKER


def test_get_angles():
    assert 8.0 == get_angles(8, 0, 512)
    assert 8.0 == get_angles(8, 1, 512)
    assert 7.717292959289593 == get_angles(8, 2, 512)
    assert 7.717292959289593 == get_angles(8, 3, 512)
    assert 6.927714586880523 == get_angles(8, 8, 512)


def test_positional_encoding():
    x = positional_encoding(64, 512).numpy()
    assert (1, 64, 512) == x.shape
    assert float(0.0998334139585495) == float(x[0][10][256])


def test_shapes():
    x = biobeaker.ffn(32, 128)
    y = x(np.random.rand(5, 1024))
    assert (5, 32) == y.shape

    x = biobeaker.EncoderLayer(32, 32, 2, 0.10, 0.10, tf.nn.swish)
    y = x(np.random.rand(1, 5, 32))

    assert (1, 5, 32) == y[0].shape
    assert (1, 2, 5, 5) == y[1].shape

    x = biobeaker.Encoder(2, 64, 128, 2, 64)
    y = x(np.random.rand(5, 5, 64))

    assert (5, 5, 128) == y[0].shape
    assert 2 == len(y[1].keys())
    assert 2 == len(y[2])

    x = biobeaker.BEAKER(4, 64, 128, 4, 128, 256)
    y = x(np.random.rand(5, 5, 64))

    assert (5, 5, 128) == y[0].shape
    assert 4 == len(y[1].keys())
    assert (4, 5, 5, 128) == np.asarray(y[2]).shape


def test_kmer_str_fns():
    x = calc_kmer_numeric_tuple(21, 98432891)
    assert (21,) == np.asarray(x).shape

    y = convert_tuple_to_string(x)
    assert "AAAAAAAAANAATGGCNCACT" == y

    y = convert_tuple_to_np(21, x)
    assert (105,) == y.shape
    assert 21.0 == np.sum(y)

    y = convert_tuple_to_string(x)
    y = convert_string_to_nparray(y)

    assert 21 == np.sum(
        np.equal([0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 4, 4, 3, 2, 3, 0, 3, 1], y)
    )

    y = convert_tuple_to_string(x)
    y = convert_string_to_nparray_tuple(21, y)

    assert (105,) == y.shape
    assert 21.0 == np.sum(y)


def test_beaker_masking():
    num_layers = 2
    embedding_dims = 64
    output_dims = 32
    num_heads = 8
    intermediate_dims = 64
    max_positions = 256
    dropout = 0.1
    attention_dropout = 0.1
    positional_encoding_dims = 16

    beaker = BEAKER(
        num_layers=num_layers,
        embedding_dims=embedding_dims,
        output_dims=output_dims,
        num_heads=num_heads,
        intermediate_dims=intermediate_dims,
        max_positions=max_positions,
        dropout=dropout,
        attention_dropout=attention_dropout,
        positional_encoding_dims=positional_encoding_dims,
    )

    input_data = np.random.rand(1, 10, embedding_dims).astype(np.float32)
    input_data[0, 4:] = 0.0
    input_data = tf.constant(input_data)

    mask = np.ones((1, 10), dtype=bool)
    mask[0, 4:] = False
    mask = tf.constant(mask)

    enc_output, attention_weights, all_outputs = beaker(input_data, mask=mask)

    assert np.allclose(
        enc_output.numpy()[:, 4:], 0.0
    ), "Masking failed: output does not contain zeros where expected."


@pytest.mark.parametrize("mask_value", [0.0, 1.0])
def test_beaker_masking_with_mask_value(mask_value):
    num_layers = 2
    embedding_dims = 64
    output_dims = 32
    num_heads = 8
    intermediate_dims = 64
    max_positions = 256
    dropout = 0.1
    attention_dropout = 0.1
    positional_encoding_dims = 16

    beaker = BEAKER(
        num_layers=num_layers,
        embedding_dims=embedding_dims,
        output_dims=output_dims,
        num_heads=num_heads,
        intermediate_dims=intermediate_dims,
        max_positions=max_positions,
        dropout=dropout,
        attention_dropout=attention_dropout,
        positional_encoding_dims=positional_encoding_dims,
    )

    input_data = np.random.rand(1, 10, embedding_dims).astype(np.float32)
    input_data[0, 4:] = mask_value
    input_data = tf.constant(input_data)

    mask = np.ones((1, 10), dtype=bool)
    mask[0, 4:] = False
    mask = tf.constant(mask)

    enc_output, attention_weights, all_outputs = beaker(input_data, mask=mask)

    assert np.allclose(
        enc_output.numpy()[:, 4:], 0.0
    ), f"Masking failed with mask_value {mask_value}: output does not contain zeros where expected."
