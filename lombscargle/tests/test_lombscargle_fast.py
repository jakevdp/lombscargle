from __future__ import division
import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from .. import lombscargle_fast, lombscargle_slow


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


@pytest.mark.parametrize('center_data', [True, False])
@pytest.mark.parametrize('fit_bias', [True, False])
def test_lombscargle_fast(center_data, fit_bias, data):
    t, y, dy = data
    kwds = dict(center_data=center_data, fit_bias=fit_bias)
    f0 = 0.8
    df = 0.01
    N = 40

    freq1, P1 = lombscargle_fast(t, y, dy, use_fft=True,
                                 f0=f0, df=df, Nf=N, **kwds)
    freq2, P2 = lombscargle_fast(t, y, dy, use_fft=False,
                                 f0=f0, df=df, Nf=N, **kwds)
    P3 = lombscargle_slow(t, y, dy=dy, freq=freq1, **kwds)

    assert_allclose(freq1, freq2)
    assert_allclose(P1, P2, atol=0.005)
    assert_allclose(P1, P3, atol=0.005)
