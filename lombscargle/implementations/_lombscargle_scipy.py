from __future__ import print_function, division

import numpy as np

try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def lombscargle_scipy(t, y, dy, freq, normalization='normalized',
                      center_data=True):
    """Lomb-Scargle Periodogram

    This is a wrapper of ``scipy.signal.lombscargle`` for computation of the
    Lomb-Scargle periodogram. This is a relatively fast version of the naive
    O[N^2] algorithm, but cannot handle heteroskedastic errors.

    Parameters
    ----------
    t, y, dy : array_like
        times, values, and errors of the data points. These should be
        broadcastable to the same shape.
    freq : array_like
        frequencies (not angular frequencies) at which to calculate periodogram
    normalization : string (optional, default='normalized')
        Normalization to use for the periodogram
        TODO: figure out what options to use
    center_data : bool (optional, default=True)
        if True, pre-center the data by subtracting the weighted mean
        of the input data.
    """
    if not HAS_SCIPY:
        raise ValueError("scipy must be installed to use lombscargle_scipy")

    if dy is not None:
        dy = np.ravel(np.asarray(dy))
        if dy.size != 1:
            if not np.allclose(dy[0], dy):
                raise ValueError("scipy method only supports constant errors")

    t, y = np.broadcast_arrays(t, y)
    freq = np.asarray(freq)
    assert t.ndim == 1
    assert freq.ndim == 1

    if center_data:
        y = y - y.mean()

    # Note: scipy input accepts angular frequencies
    p = signal.lombscargle(t, y, 2 * np.pi * freq)

    if normalization == 'unnormalized':
        pass
    elif normalization == 'normalized':
        p *= 2 / (t.size * np.mean(y ** 2))
    else:
        raise ValueError("normalization='{0}' "
                         "not recognized".format(normalization))
    return p
