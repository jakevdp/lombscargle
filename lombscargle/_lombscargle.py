import numpy as np

def lomb_scargle(t, y, dy, freq,
                 normalization='normalized',
                 generalized=True, subtract_mean=True):
    """
    Lomb-scargle Periodogram

    Parameters
    ----------
    t : array_like
        sequence of times
    y : array_like
        sequence of observations
    dy : array_like
        sequence of observational errors
    freq : tuple (f0, df, N)
        parameters describing the frequencies at which to compute the
        periodogram. frequencies are ``freq = f0 + df * np.arange(N)``
    normalization : string (optional, default='normalized')
        Normalization to use for the periodogram.
    fit_mean : bool (optional, default=True)
        if True, fit the data mean as part of the model
    subtract_mean : bool (optional, default=True)
        if True, pre-center the data by subtracting the weighted mean
        of the inputs.
    significance : None or float or ndarray
        if specified, then this is a list of significances to compute
        for the results.
    Returns
    -------
    p : array_like
        Lomb-Scargle power associated with each frequency omega
    """
    pass
