import pytest
import numpy as np
from numpy.testing import assert_allclose
from functools import partial

from .. import lombscargle_slow, lombscargle_scipy, lombscargle_matrix

METHODS = [partial(lombscargle_slow, fit_bias=False),
           partial(lombscargle_matrix, fit_bias=False),
           lombscargle_scipy]


@pytest.fixture
def test_data():
    rand = np.random.RandomState(42)
    t = 100 * rand.rand(100)
    y = 0.5 + np.sin(t) + 0.1 * rand.randn(100)
    freq = 0.1 * np.arange(1, 100)
    return t, y, freq


@pytest.mark.parametrize('lombscargle_method', METHODS)
@pytest.mark.parametrize('center_data', [True, False])
@pytest.mark.parametrize('normalization', ['normalized', 'unnormalized'])
def test_lombscargle_methods_common(lombscargle_method, center_data,
                                    normalization, test_data):
    t, y, freq = test_data
    kwds = dict(normalization=normalization, center_data=center_data)

    expected_output = lombscargle_slow(t, y, freq, dy=np.ones_like(t),
                                       fit_bias=False, **kwds)

    output = lombscargle_method(t, y, freq, **kwds)
    assert_allclose(output, expected_output)
