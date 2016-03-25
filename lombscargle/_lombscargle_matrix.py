from __future__ import print_function, division

import numpy as np


def lombscargle_matrix(t, y, freq, dy=1, normalization='normalized',
                       fit_bias=True, center_data=True):
    """Lomb-Scargle Periodogram

    This is a pure python matrix-based implementation which is relatively slow,
    but useful for validating the faster algorithms in the package.

    Parameters
    ----------
    t : array_like
        sequence of times
    y : array_like
        sequence of observations
    freq : array_like
        frequencies (not angular frequencies) at which to calculate periodogram
    dy : float or array_like (optional)
        sequence of observational errors
    normalization : string (optional, default='normalized')
        Normalization to use for the periodogram
        TODO: figure out what options to use
    fit_bias : bool (optional, default=True)
        if True, include a constant offet as part of the model at each
        frequency. This can lead to more accurate results, especially in then
        case of incomplete phase coverage.
    center_data : bool (optional, default=True)
        if True, pre-center the data by subtracting the weighted mean
        of the input data. This is especially important if ``fit_bias = False``

    Returns
    -------
    power : array_like
        Lomb-Scargle power associated with each frequency omega.
        Units of the result depend on the normalization.
    """
    t, y, dy = np.broadcast_arrays(t, y, dy)
    freq = np.asarray(freq)
    assert t.ndim == 1
    assert freq.ndim == 1

    w = dy ** -2.0
    w /= w.sum()

    # if fit_bias is true, centering the data now simplifies the math below.
    if center_data or fit_bias:
        yw = (y - np.dot(w, y)) / dy
    else:
        yw = y / dy
    chi2_ref = np.dot(yw, yw)

    # compute the model chi2 at each frequency
    def compute_power(f):
        cols = [np.ones_like(t)] if fit_bias else []
        cols.append(np.sin(2 * np.pi * f * t))
        cols.append(np.cos(2 * np.pi * f * t))
        X = np.transpose(np.vstack(cols) / dy)
        XTX = np.dot(X.T, X)
        XTy = np.dot(X.T, yw)
        return np.dot(XTy.T, np.linalg.solve(XTX, XTy))

    p = np.array([compute_power(f) for f in freq])

    if normalization == 'unnormalized':
        p *= 0.5
    elif normalization == 'normalized':
        p /= chi2_ref
    else:
        raise ValueError("normalization='{0}' "
                         "not recognized".format(normalization))
    return p
