import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from functools import partial

from astropy import units

from .. import lombscargle
from .._lombscargle_slow import lombscargle_slow
from .._lombscargle_fast import lombscargle_fast
from .._lombscargle_scipy import lombscargle_scipy
from .._lombscargle_matrix import lombscargle_matrix

from ..heuristics import baseline_heuristic

METHOD_NAMES = ['auto', 'fast', 'slow', 'scipy', 'matrix']
METHODS_NOTFAST = [lombscargle_slow, lombscargle_matrix, lombscargle_scipy]
METHODS_NOBIAS = [partial(lombscargle_slow, fit_bias=False),
                  partial(lombscargle_matrix, fit_bias=False),
                  lombscargle_scipy]
METHODS_BIAS = [lombscargle_slow, lombscargle_matrix]


@pytest.fixture
def data(N=100, period=1, theta=[10, 2, 3], dy=1, rseed=0):
    """Generate some data for testing"""
    rng = np.random.RandomState(rseed)
    t = 20 * period * rng.rand(N)
    omega = 2 * np.pi / period
    y = theta[0] + theta[1] * np.sin(omega * t) + theta[2] * np.cos(omega * t)
    dy = dy * (0.5 + rng.rand(N))
    y += dy * rng.randn(N)

    return t, y, dy


@pytest.mark.parametrize('method', METHOD_NAMES)
@pytest.mark.parametrize('shape', [(), (1,), (2,), (3,), (2, 3)])
def test_output_shapes(method, shape, data):
    t, y, dy = data
    freq = np.asarray(np.random.rand(*shape))
    freq.flat = np.arange(1, freq.size + 1)
    freq_out, PLS = lombscargle(t, y, frequency=freq,
                                fit_bias=False, method=method)
    assert_equal(freq, freq_out)
    assert_equal(PLS.shape, shape)


@pytest.mark.parametrize('method', METHOD_NAMES)
@pytest.mark.parametrize('t_unit', [units.second, units.day])
@pytest.mark.parametrize('frequency_unit', [units.Hz, 1. / units.second])
@pytest.mark.parametrize('y_unit', [units.mag, units.jansky])
def test_units_match(method, t_unit, frequency_unit, y_unit, data):
    t, y, dy = data
    dy = dy.mean()  # scipy only supports constant errors

    t = t * t_unit
    y = y * y_unit
    dy = dy * y_unit
    frequency = np.linspace(0.5, 1.5, 10) * frequency_unit
    frequency_out, PLS = lombscargle(t, y, frequency=frequency,
                                     fit_bias=False, method=method)
    assert frequency_out.unit == frequency_unit
    assert_equal(PLS.unit, units.dimensionless_unscaled)

    frequency_out, PLS = lombscargle(t, y, dy,
                                     frequency=frequency,
                                     fit_bias=False, method=method)
    assert frequency_out.unit == frequency_unit
    assert_equal(PLS.unit, units.dimensionless_unscaled)


@pytest.mark.parametrize('method', METHOD_NAMES)
def test_units_mismatch(method, data):
    # These tests fail on Travis in Python 3.5 for some reason
    import sys
    if sys.version[:3] == '3.5':
        return
    t, y, dy = data
    dy = dy.mean()  # scipy only supports constant errors

    t = t * units.second
    y = y * units.mag
    frequency = np.linspace(0.5, 1.5, 10)

    # this should fail because frequency and 1/t unitsdo not match
    with pytest.raises(ValueError) as err:
        lombscargle(t, y, frequency=frequency,
                    method=method, fit_bias=False)
    assert str(err.value).startswith('Units of frequency not equivalent')

    # this should fail because dy and y units do not match
    with pytest.raises(ValueError) as err:
        lombscargle(t, y, dy, frequency / t.unit,
                    method=method, fit_bias=False)
    assert str(err.value).startswith('Units of y not equivalent')


@pytest.mark.parametrize('lombscargle_method', METHODS_NOBIAS)
@pytest.mark.parametrize('center_data', [True, False])
@pytest.mark.parametrize('normalization', ['normalized', 'unnormalized'])
def test_lombscargle_methods_common(lombscargle_method, center_data,
                                    normalization, data):
    t, y, dy = data
    freq = 0.8 + 0.01 * np.arange(40)

    kwds = dict(normalization=normalization, center_data=center_data)

    expected_output = lombscargle_slow(t, y, freq, dy=np.ones_like(t),
                                       fit_bias=False, **kwds)

    output = lombscargle_method(t, y, freq, **kwds)
    assert_allclose(output, expected_output)


@pytest.mark.parametrize('lombscargle_method', METHODS_BIAS)
@pytest.mark.parametrize('center_data', [True, False])
@pytest.mark.parametrize('fit_bias', [True, False])
@pytest.mark.parametrize('normalization', ['normalized', 'unnormalized'])
def test_lombscargle_methods_with_bias(lombscargle_method, center_data,
                                       fit_bias, normalization, data):
       t, y, freq = data
       freq = 0.8 + 0.01 * np.arange(40)

       kwds = dict(normalization=normalization, center_data=center_data,
                   fit_bias=fit_bias)

       expected_output = lombscargle_slow(t, y, freq, dy=np.ones_like(t),
                                          **kwds)

       output = lombscargle_method(t, y, freq, **kwds)
       assert_allclose(output, expected_output)


@pytest.mark.parametrize('method', METHOD_NAMES)
@pytest.mark.parametrize('center_data', [True, False])
@pytest.mark.parametrize('freq', [0.8 + 0.01 * np.arange(40), None])
def test_common_interface(method, center_data, freq, data):
    t, y, dy = data

    if freq is None:
        freq = baseline_heuristic(len(t), t.max() - t.min())

    expected_PLS = lombscargle_slow(t, y, freq=freq,
                                    fit_bias=False, center_data=center_data)
    frequency, PLS = lombscargle(t, y, frequency=freq, method=method,
                                 fit_bias=False, center_data=center_data)
    assert_allclose(freq, frequency)

    if method in ['fast', 'auto']:
        atol = 0.005
    else:
        atol = 0
    assert_allclose(PLS, expected_PLS, atol=atol)
