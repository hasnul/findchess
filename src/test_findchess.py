from findchess import Line, Contours, Quadrangle
from numpy.testing import assert_allclose


def test_instantiate_line():
    rho, theta = 0.0, 0.0
    line = Line(rho, theta)
    assert line._rho == rho
    assert line._theta == theta
    assert_allclose(line._cos_factor, 1.0)
    assert_allclose(line._sin_factor, 0.0)
    assert_allclose(line._center, (0.0, 0.0))
