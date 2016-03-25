from __future__ import division

import numpy as np


def get_heuristic(heuristic):
    if heuristic == 'baseline':
        return baseline_heuristic
    else:
        raise ValueError("Unrecognized heuristic '{0}'".format(heuristic))


def baseline_heuristic(n_samples, baseline,
                       samples_per_peak=5,
                       nyquist_factor=5,
                       minimum_frequency=None,
                       maximum_frequency=None):
    """Use a heuristic to compute a frequency grid.

    Note that this assumes that the baseline is much larger than the
    oscillation period, such that the peak widths are driven by the baseline.
    If you are searching for periods longer than the baseline of your
    observations, this will not perform well.

    Even with a large baseline, be aware that this heuristic is based on the
    concept of "average Nyquist frequency", which is fundamentally incorrect
    for irregularly-sampled data. The `nyquist_factor` provides a fudge factor
    that allows sampling of higher frequencies.

    Parameters
    ----------
    n_samples : int
        the number of samples in your dataset. This is used to guess at the
        maximum frequency if ``maximum_frequency`` is not provided.
    baseline : float
        the difference between the maximum and minimum observation times.
        This is used to estimate the peak width, and thus choose the frequency
        spacing.
    samples_per_peak : float (optional, default=5)
        The approximate number of desired samples across the typical peak width
    nyquist_factor : float (optional, default=5)
        The multiple of the average nyquist frequency used to choose the
        maximum frequency if ``maximum_frequency`` is not provided.
    minimum_frequency : float (optional)
        If specified, then use this minimum frequency rather than one chosen
        based on the size of the baseline.
    maximum_frequency : float (optional)
        If specified, then use this maximum frequency rather than one chosen
        based on the average nyquist frequency.
    return_tuple : bool (optional, default=False)
        if True, return a tuple `(freq_min, freq_spacing, N_freq)`. Otherwise,
        return an array `freq_min + freq_spacing * np.arange(N_freq)`

    Returns
    -------
    freq : ndarray or tuple
        If `return_tuple` is True, a tuple `(freq_min, freq_spacing, N_freq)`
        otherwise, an array `freq_min + freq_spacing * np.arange(N_freq)`.
    """
    df = 1. / baseline / oversampling

    if minimum_frequency is not None:
        f0 = minimum_frequency
    else:
        f0 = 0.5 * df

    if maximum_frequency is not None:
        Nf = int(np.ceil((maximum_frequency - f0) / df))
    else:
        Nf = int(0.5 * oversampling * nyquist_factor * n_samples)

    return f0 + df * np.arange(Nf)
