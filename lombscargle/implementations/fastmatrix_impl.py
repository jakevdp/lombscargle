from __future__ import print_function, division

import numpy as np

from .utils import trig_sum
from .mle import design_matrix


def lombscargle_fastmatrix(t, y, dy, frequency, normalization='normalized',
                           fit_bias=True, center_data=True, nterms=1,
                           use_fft=True, trig_sum_kwds=None):
    """Lomb-Scargle Periodogram

    This implements a fast matrix-based periodogram using the Fast-chi-square
    algorithm outlined in [4]_. The key advantage of this algorithm is the
    ability to compute multiterm periodograms relatively quickly.

    Parameters
    ----------
    t, y, dy : array_like
        times, values, and errors of the data points. These should be
        broadcastable to the same shape.
    frequency : array_like
        frequencies (not angular frequencies) at which to calculate periodogram
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
    nterms : int (optional, default=1)
        Number of Fourier terms in the fit

    Returns
    -------
    power : array_like
        Lomb-Scargle power associated with each frequency.
        Units of the result depend on the normalization.

    References
    ----------
    .. [1] M. Zechmeister and M. Kurster, A&A 496, 577-584 (2009)
    .. [2] W. Press et al, Numerical Recipies in C (2002)
    .. [3] Scargle, J.D. ApJ 263:835-853 (1982)
    .. [4] Palmer, J. ApJ 695:496–502 (2009)
    """
    if nterms == 0 and not fit_bias:
        raise ValueError("Cannot have nterms = 0 without fitting bias")

    if dy is None:
        dy = 1

    t, y, dy = np.broadcast_arrays(t, y, dy)
    frequency = np.asarray(frequency)

    assert t.ndim == 1
    assert frequency.ndim == 1

    # Get frequency grid
    f0 = frequency[0]
    df = frequency[1] - frequency[0]
    Nf = len(frequency)
    assert np.allclose(frequency, f0 + df * np.arange(Nf))

    w = dy ** -2.0
    ws = np.sum(w)

    # if fit_bias is true, centering the data now simplifies the math below.
    if center_data or fit_bias:
        y = y - np.dot(w, y) / ws

    yw = y / dy
    chi2_ref = np.dot(yw, yw)

    kwargs = dict.copy(trig_sum_kwds or {})
    kwargs.update(f0=f0, df=df, use_fft=use_fft, N=Nf)

    # Here we build-up the matrices XTX and XTy using pre-computed
    # sums. The relevant identities are
    # 2 sin(mx) sin(nx) = cos(m−n)x − cos(m+n)x
    # 2 cos(mx) cos(nx) = cos(m−n)x + cos(m+n)x
    # 2 sin(mx) cos(nx) = sin(m−n)x + sin(m+n)x

    yws = np.sum(y * w)

    SCw = [(np.zeros(Nf), ws * np.ones(Nf))]
    SCw.extend([trig_sum(t, w, freq_factor=i, **kwargs)
                for i in range(1, 2 * nterms + 1)])
    Sw, Cw = zip(*SCw)

    SCyw = [(np.zeros(Nf), yws * np.ones(Nf))]
    SCyw.extend([trig_sum(t, w * y, freq_factor=i, **kwargs)
                 for i in range(1, nterms + 1)])
    Syw, Cyw = zip(*SCyw)

    # Now create an indexing scheme so we can quickly
    # build-up matrices at each frequency
    order = [('C', 0)] if fit_bias else []
    order.extend(sum([[('S', i), ('C', i)]
                      for i in range(1, nterms + 1)], []))

    funcs = dict(S=lambda m, i: Syw[m][i],
                 C=lambda m, i: Cyw[m][i],
                 SS=lambda m, n, i: 0.5 * (Cw[abs(m - n)][i] - Cw[m + n][i]),
                 CC=lambda m, n, i: 0.5 * (Cw[abs(m - n)][i] + Cw[m + n][i]),
                 SC=lambda m, n, i: 0.5 * (np.sign(m - n) * Sw[abs(m - n)][i]
                                           + Sw[m + n][i]),
                 CS=lambda m, n, i: 0.5 * (np.sign(n - m) * Sw[abs(n - m)][i]
                                           + Sw[n + m][i]))

    def compute_power(i):
        XTX = np.array([[funcs[A[0] + B[0]](A[1], B[1], i)
                         for A in order]
                        for B in order])
        XTy = np.array([funcs[A[0]](A[1], i) for A in order])
        return np.dot(XTy.T, np.linalg.solve(XTX, XTy))

    p = np.array([compute_power(i) for i in range(Nf)])

    if normalization == 'unnormalized':
        p *= 0.5
    elif normalization == 'normalized':
        p /= chi2_ref
    else:
        raise ValueError("normalization='{0}' "
                         "not recognized".format(normalization))
    return p
