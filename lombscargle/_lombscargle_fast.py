from __future__ import print_function, division

import warnings

import numpy as np
from ._utils import trig_sum


def lombscargle_fast(t, y, dy=1, f0=0, df=None, Nf=None,
                     center_data=True, fit_bias=True,
                     normalization='normalized',
                     use_fft=True, freq_oversampling=5, nyquist_factor=2,
                     trig_sum_kwds=None):
    """Compute a lomb-scargle periodogram for the given data
    This implements both an O[N^2] method if use_fft==False, or an
    O[NlogN] method if use_fft==True.
    Parameters
    ----------
    t, y, dy : array_like
        times, values, and errors of the data points. These should be
        broadcastable to the same shape. If dy is not specified, a
        constant error will be used.
    f0, df, Nf : (float, float, int)
        parameters describing the frequency grid, f = f0 + df * arange(Nf).
        Defaults, with T = t.max() - t.min():
        - f0 = 0
        - df is set such that there are ``freq_oversampling`` points per
          peak width. ``freq_oversampling`` defaults to 5.
        - Nf is set such that the highest frequency is ``nyquist_factor``
          times the so-called "average Nyquist frequency".
          ``nyquist_factor`` defaults to 2.
        Note that for unevenly-spaced data, the periodogram can be sensitive
        to frequencies far higher than the average Nyquist frequency.
    normalization : string (optional, default='normalized')
        Normalization to use for the periodogram
        TODO: figure out what options to use
    center_data : bool (default=True)
        Specify whether to subtract the mean of the data before the fit
    fit_bias : bool (default=True)
        If True, then compute the floating-mean periodogram; i.e. let the mean
        vary with the fit.
    use_fft : bool (default=True)
        If True, then use the Press & Rybicki O[NlogN] algorithm to compute
        the result. Otherwise, use a slower O[N^2] algorithm
    Other Parameters
    ----------------
    freq_oversampling : float (default=5)
        Oversampling factor for the frequency bins. Only referenced if
        ``df`` is not specified
    nyquist_factor : float (default=2)
        Parameter controlling the highest probed frequency. Only referenced
        if ``Nf`` is not specified.
    trig_sum_kwds : dict or None (optional)
        extra keyword arguments to pass to the ``trig_sum`` utility.
        Options are ``oversampling`` and ``Mfft``. See documentation
        of ``trig_sum`` for details.
    Notes
    -----
    Note that the ``use_fft=True`` algorithm is an approximation to the true
    Lomb-Scargle periodogram, and as the number of points grows this
    approximation improves. On the other hand, for very small datasets
    (<~50 points or so) this approximation may not be useful.
    References
    ----------
    .. [1] Press W.H. and Rybicki, G.B, "Fast algorithm for spectral analysis
        of unevenly sampled data". ApJ 1:338, p277, 1989
    .. [2] M. Zechmeister and M. Kurster, A&A 496, 577-584 (2009)
    .. [3] W. Press et al, Numerical Recipies in C (2002)
    """
    # Validate and setup input data
    t, y, dy = map(np.ravel, np.broadcast_arrays(t, y, dy))
    w = dy ** -2
    w /= w.sum()

    # Validate and setup frequency grid
    if df is None:
        peak_width = 1. / (t.max() - t.min())
        df = peak_width / freq_oversampling
    if Nf is None:
        avg_Nyquist = 0.5 * len(t) / (t.max() - t.min())
        Nf = max(16, (nyquist_factor * avg_Nyquist - f0) / df)
    Nf = int(Nf)
    assert(df > 0)
    assert(Nf > 0)
    freq = f0 + df * np.arange(Nf)

    # Center the data. Even if we're fitting the offset,
    # this step makes the expressions below more succinct
    if center_data or fit_bias:
        y = y - np.dot(w, y)

    # set up arguments to trig_sum
    kwargs = dict.copy(trig_sum_kwds or {})
    kwargs.update(f0=f0, df=df, use_fft=use_fft, N=Nf)

    #----------------------------------------------------------------------
    # 1. compute functions of the time-shift tau at each frequency
    Sh, Ch = trig_sum(t, w * y, **kwargs)
    S2, C2 = trig_sum(t, w, freq_factor=2, **kwargs)

    if fit_bias:
        S, C = trig_sum(t, w, **kwargs)
        with warnings.catch_warnings():
            # Filter "invalid value in divide" warnings for zero-frequency
            if f0 == 0:
                warnings.simplefilter("ignore")
            tan_2omega_tau = (S2 - 2 * S * C) / (C2 - (C * C - S * S))
            # fix NaN at zero frequency
            if np.isnan(tan_2omega_tau[0]):
                tan_2omega_tau[0] = 0
    else:
        tan_2omega_tau = S2 / C2

    # slower/less stable way: we'll use trig identities instead
    # omega_tau = 0.5 * np.arctan(tan_2omega_tau)
    # S2w, C2w = np.sin(2 * omega_tau), np.cos(2 * omega_tau)
    # Sw, Cw = np.sin(omega_tau), np.cos(omega_tau)

    S2w = tan_2omega_tau / np.sqrt(1 + tan_2omega_tau * tan_2omega_tau)
    C2w = 1 / np.sqrt(1 + tan_2omega_tau * tan_2omega_tau)
    Cw = np.sqrt(0.5) * np.sqrt(1 + C2w)
    Sw = np.sqrt(0.5) * np.sign(S2w) * np.sqrt(1 - C2w)

    #----------------------------------------------------------------------
    # 2. Compute the periodogram, following Zechmeister & Kurster
    #    and using tricks from Press & Rybicki.
    YY = np.dot(w, y ** 2)
    YC = Ch * Cw + Sh * Sw
    YS = Sh * Cw - Ch * Sw
    CC = 0.5 * (1 + C2 * C2w + S2 * S2w)
    SS = 0.5 * (1 - C2 * C2w - S2 * S2w)

    if fit_bias:
        CC -= (C * Cw + S * Sw) ** 2
        SS -= (S * Cw - C * Sw) ** 2

    with warnings.catch_warnings():
        # Filter "invalid value in divide" warnings for zero-frequency
        if fit_bias and f0 == 0:
            warnings.simplefilter("ignore")

        power = (YC * YC / CC + YS * YS / SS)

        # fix NaN and INF at zero frequency
        if np.isnan(power[0]) or np.isinf(power[0]):
            power[0] = 0

    if normalization == 'normalized':
        power /= YY
    elif normalization == 'unnormalized':
        power *= 0.5 * t.size
    else:
        raise ValueError("normalization='{0}' "
                         "not recognized".format(normalization))

    return freq, power
