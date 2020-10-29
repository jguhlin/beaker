import tensorflow as tf
import numpy as np

import biobeaker
from biobeaker.utils import get_angles, positional_encoding

def test_get_angles():
    assert(8.0 == get_angles(8, 0, 512))
    assert(8.0 == get_angles(8, 1, 512))
    assert(7.717292959289593 == get_angles(8, 2, 512))
    assert(7.717292959289593 == get_angles(8, 3, 512))
    assert(6.927714586880523 == get_angles(8, 8, 512))

def test_positional_encoding():
    x = positional_encoding(64, 512).numpy()
    assert((1, 64, 512) == x.shape)
    assert(float(0.0998334139585495) == float(x[0][10][256]))

def test_shapes():
    x = biobeaker.ffn(32, 128)
    y = x(np.random.rand(5, 1024))
    assert((5, 32) == y.shape)

    x = biobeaker.EncoderLayer(32, 2, 0.10, 0.10, tf.nn.swish)
    y = x(np.random.rand(5, 32))

    assert((5, 32) == y[0].shape)
    assert((2, 5, 5) == y[1].shape)

    x = biobeaker.Encoder(2, 64, 128, 2, 64)
    y = x(np.random.rand(5, 5, 64))

    assert((5,5,64) == y[0].shape)
    assert(2 == len(y[1].keys()))
    assert(2 == len(y[2]))

    x = biobeaker.BEAKER(4, 64, 128, 4, 128, 256)
    y = x(np.random.rand(5,5,64))

    assert((5,5,128) == y[0].shape)
    assert(4 == len(y[1].keys()))
    assert((4, 5, 5, 128) == np.asarray(y[2]).shape)


