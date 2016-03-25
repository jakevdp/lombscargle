from __future__ import print_function, division

import numpy as np

try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def lombscargle_scipy(t, y, freq, normalization='normalized',
                      center_data=True):
    """Lomb Scargle Periodogram computed via scipy

    Parameters
    ----------
    t : array_like
        sequence of times
    y : array_like
        sequence of observations
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
        raise ValueError("scipy not available")

    t, y = np.broadcast_arrays(t, y)

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
