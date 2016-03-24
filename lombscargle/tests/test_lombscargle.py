import pytest
import numpy as np
from numpy.testing import assert_allclose
from functools import partial

from .. import lombscargle_slow, lombscargle_scipy, lombscargle_matrix

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


@pytest.mark.parametrize('lombscargle_method', METHODS_NOBIAS)
@pytest.mark.parametrize('center_data', [True, False])
@pytest.mark.parametrize('normalization', ['normalized', 'unnormalized'])
def test_lombscargle_methods_common(lombscargle_method, center_data,
                                    normalization, data):
    t, y, freq = data
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
       kwds = dict(normalization=normalization, center_data=center_data,
                   fit_bias=fit_bias)

       expected_output = lombscargle_slow(t, y, freq, dy=np.ones_like(t),
                                          **kwds)

       output = lombscargle_method(t, y, freq, **kwds)
       assert_allclose(output, expected_output)
