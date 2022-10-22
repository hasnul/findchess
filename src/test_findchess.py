from findchess import Line, Contours, LineError, Quadrangle
import numpy as np
from numpy.testing import assert_allclose
import pytest

BOX_WIDTH = 100  # pixels
assert BOX_WIDTH % 2 == 0


@pytest.fixture
def line_params():
    return {'rho': np.sqrt(BOX_WIDTH**2 + BOX_WIDTH**2)/2, 'theta': np.pi/4}


@pytest.fixture
def line(line_params):
    rho = line_params['rho']
    theta = line_params['theta']
    return Line(rho, theta)


def test_instantiate_line(line, line_params):
    rho = line_params['rho']
    theta = line_params['theta']
    assert line._rho == rho
    assert line._theta == theta
    assert_allclose(line._cos_factor, np.cos(theta))
    assert_allclose(line._sin_factor, np.sin(theta))
    assert_allclose(line._center, (np.cos(theta)*rho, np.sin(theta)*rho))


def test_invalid_line_params():
    with pytest.raises(LineError):
        Line(-1.0, 0.0)
    with pytest.raises(LineError):
        Line(1.0, -1.0)
    with pytest.raises(LineError):
        Line(1.0, np.pi + 0.1)


def test_line_getters(line):
    assert line.get_rho() == line._rho
    assert line.get_theta() == line._theta
    assert_allclose(line.get_center(), line._center)


def test_line_get_segment(line, line_params):
    with pytest.raises(LineError):
        line.get_segment(-100, 100)
    with pytest.raises(LineError):
        line.get_segment(1, -50)

    line1 = Line(BOX_WIDTH//2, 0)
    p1, p2 = line1.get_segment(BOX_WIDTH, BOX_WIDTH)
    assert p1 == (BOX_WIDTH//2, BOX_WIDTH) and p2 == (BOX_WIDTH//2, -BOX_WIDTH)

    line2 = Line(BOX_WIDTH//2, np.pi/2)
    p1, p2 = line2.get_segment(BOX_WIDTH, BOX_WIDTH)
    assert p1 == (-BOX_WIDTH, BOX_WIDTH//2) and p2 == (BOX_WIDTH, BOX_WIDTH//2)

    rho = line_params['rho']
    p1, p2 = line.get_segment(rho, rho)
    assert p1 == (0, BOX_WIDTH) and p2 == (BOX_WIDTH, 0)


def test_line_bool_methods(line, line_params):
    theta = line_params['theta']
    assert line.is_below_threshold(theta + 0.2*theta)
    # assert line1.is_below_threshold(theta)  # Is this a reliable test?
    assert line.is_above_threshold(theta - 0.5*theta)


def test_line_intersect(line, line_params):
    with pytest.raises(LineError):
        line1 = Line(50, np.pi/3)
        line2 = Line(70, np.pi/3)
        line2.intersect(line1)

    theta = line_params['theta']
    intersecting_line = Line(0, np.pi - theta)
    x, y = line.intersect(intersecting_line)
    assert round(x) == BOX_WIDTH//2 and round(y) == BOX_WIDTH//2


def test_line_draw(line):
    blank = np.zeros((BOX_WIDTH, BOX_WIDTH, 3), dtype=np.uint8)
    assert np.all(blank == 0)
    blank_canvas = blank.copy()
    line.draw(blank_canvas)
    assert not np.all(blank_canvas == 0)

    BLACK = [0, 0, 0]
    RED = [0, 0, 255]
    GREEN = [0, 255, 0]

    line1 = Line(50, 0)
    line2 = Line(50, np.pi/2)
    blank_canvas = blank.copy()
    line1.draw(blank_canvas, color=tuple(RED), thickness=1)
    line2.draw(blank_canvas, color=tuple(GREEN), thickness=1)
    rows = slice(49, 52)
    assert np.all(blank_canvas[rows, 49] == np.array([BLACK, GREEN, BLACK]))
    assert np.all(blank_canvas[rows, 50] == np.array([RED, GREEN, RED]))
    assert np.all(blank_canvas[rows, 51] == np.array([BLACK, GREEN, BLACK]))


def test_line_repr(line):
    assert isinstance(line.__repr__(), str)
    repr = f"(t: {line._theta*180/np.pi:.2f}deg, r: {line._rho:.0f})"
    assert line.__repr__() == repr
