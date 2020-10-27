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
