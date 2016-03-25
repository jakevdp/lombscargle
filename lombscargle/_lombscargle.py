import numpy as np


def lombscargle(t, y, dy=None,
                frequency=None,
                method='auto',
                assume_regular_frequency=False,
                normalization='normalized',
                fit_bias=True, center_data=True,
                frequency_heuistic='baseline'):
    """
    Lomb-scargle Periodogram

    Parameters
    ----------
    t : array_like or Quantity
        sequence of observation times
    y : array_like or Quantity
        sequence of observations associated with times t
    dy : float, array_like or Quantity (optional)
        sequence of observational errors associated with times t
    frequency : array (optional)
        frequencies (not angular frequencies) at which to evaluate the
        periodogram. If not specified, optimal frequencies will be chosen using
        a heuristic which will attempt to provide sufficient frequency range
        and sampling so that peaks will not be missed. Note that in order to
        use method='fast', frequencies must be regularly spaced.
    method : string (optional)
        specify the lomb scargle implementation to use. Options are:
        - 'auto': choose the best method based on the input
        - 'fast': use the O[N log N] fast method. Note that this requires
          evenly-spaced frequencies: by default this will be checked unless
          `assume_regular_frequency` is set to True.
        - `slow`: use the O[N^2] pure python implementation
        - `matrix`: use the O[N^2] matrix/linear-fitting implementation
    assume_regular_frequency : bool (optional)
        if True, assume that the input frequency is of the form
        freq = f0 + df * np.arange(N). Only referenced if method is 'auto'
        or 'fast'.
    normalization : string (optional, default='normalized')
        Normalization to use for the periodogram. Options are 'normalized' or
        'unnormalized'.
    fit_bias : bool (optional, default=True)
        if True, fit the bias (i.e. constant data offset) as part of the model
        at each frequency.
    fit_bias : bool (optional, default=True)
        if True, include a constant offet as part of the model at each
        frequency. This can lead to more accurate results, especially in then
        case of incomplete phase coverage.
    center_data : bool (optional, default=True)
        if True, pre-center the data by subtracting the weighted mean
        of the input data. This is especially important if `fit_bias = False`
    frequency_heuristic : string (optional, default='baseline')
        the frequency heuristic to use. By default, it is assumed that the
        observation baseline will drive the peak width.
    Returns
    -------
    freq : array_like
        Frequencies at which the Lomb-Scargle power is computed
    PLS : array_like
        Lomb-Scargle power associated with each frequency omega
    """
    pass
