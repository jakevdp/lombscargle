import numpy as np


def lombscargle_slow(t, y, freq, dy=None, normalization='normalized',
                     fit_bias=True, center_data=True):
    """Lomb-Scargle Periodogram

    This is a pure python implementation which is relatively slow, but useful
    for validating the faster algorithms in the package.

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

    References
    ----------
    .. [1] M. Zechmeister and M. Kurster, A&A 496, 577-584 (2009)
    .. [2] W. Press et al, Numerical Recipies in C (2002)
    .. [3] Scargle, J.D. 1982, ApJ 263:835-853
    """
    if dy is None:
        dy = 1
    t, y, dy = np.broadcast_arrays(t, y, dy)
    assert t.ndim == 1

    freq = np.asarray(freq)
    freqshape = freq.shape

    w = dy ** -2.0
    w /= w.sum()

    if center_data:
        # subtract MLE for mean in the presence of noise.
        y = y - np.dot(w, y)

    omega = 2 * np.pi * freq
    omega = omega.ravel()[np.newaxis, :]

    # make following arrays into column vectors
    t, y, dy, w = map(lambda x: x[:, np.newaxis], (t, y, dy, w))

    sin_omega_t = np.sin(omega * t)
    cos_omega_t = np.cos(omega * t)

    # compute time-shift tau
    # S2 = np.dot(w.T, np.sin(2 * omega * t)
    S2 = 2 * np.dot(w.T, sin_omega_t * cos_omega_t)
    # C2 = np.dot(w.T, np.cos(2 * omega * t)
    C2 = 2 * np.dot(w.T, 0.5 - sin_omega_t ** 2)

    if fit_bias:
        S = np.dot(w.T, sin_omega_t)
        C = np.dot(w.T, cos_omega_t)

        S2 -= (2 * S * C)
        C2 -= (C * C - S * S)

    # compute components needed for the fit
    tan_2omega_tau = S2 / C2
    omega_t_tau = omega * t - 0.5 * np.arctan(tan_2omega_tau)

    sin_omega_t_tau = np.sin(omega_t_tau)
    cos_omega_t_tau = np.cos(omega_t_tau)

    Y = np.dot(w.T, y)
    YY = np.dot(w.T, y * y) - Y * Y

    wy = w * y

    YCtau = np.dot(wy.T, cos_omega_t_tau)
    YStau = np.dot(wy.T, sin_omega_t_tau)
    CCtau = np.dot(w.T, cos_omega_t_tau * cos_omega_t_tau)
    SStau = np.dot(w.T, sin_omega_t_tau * sin_omega_t_tau)

    if fit_bias:
        Ctau = np.dot(w.T, cos_omega_t_tau)
        Stau = np.dot(w.T, sin_omega_t_tau)

        YCtau -= Y * Ctau
        YStau -= Y * Stau
        CCtau -= Ctau * Ctau
        SStau -= Stau * Stau

    p_omega = (YCtau * YCtau / CCtau + YStau * YStau / SStau) / YY
    return p_omega.reshape(freqshape)
