from __future__ import division

import numpy as np


class Heuristic(object):
    """Base class for heuristics to choose appropriate frequency grids"""
    def frequency_grid(self, t, y, dy, **kwargs):
        raise NotImplementedError()

    @classmethod
    def get(cls, heuristic):
        if heuristic == 'baseline':
            return BaselineHeuristic()
        else:
            raise ValueError("Unrecognized heuristic '{0}'".format(heuristic))


class BaselineHeuristic(Heuristic):
    """
    Compute a frequency grid assuming that peak width is driven
    by the observational baseline.

    Note that this assumes that the baseline is much larger than the
    oscillation period, such that the peak widths are driven by the baseline.
    If you are searching for periods longer than the baseline of your
    observations, this will not perform well.

    Even with a large baseline, be aware that this heuristic is based on the
    concept of "average Nyquist frequency", which is fundamentally incorrect
    for irregularly-sampled data. The `nyquist_factor` provides a fudge factor
    that allows sampling of higher frequencies.
    """
    def frequency_grid(self, t, y=None, dy=None,
                       samples_per_peak=5,
                       nyquist_factor=5,
                       minimum_frequency=None,
                       maximum_frequency=None):
        """Use a heuristic to compute a frequency grid.

        Parameters
        ----------
        t, y, dy : arrays
            data arrays. Only t is referenced here.
        samples_per_peak : float (optional, default=5)
            The approximate number of desired samples across the typical peak
        nyquist_factor : float (optional, default=5)
            The multiple of the average nyquist frequency used to choose the
            maximum frequency if ``maximum_frequency`` is not provided.
        minimum_frequency : float (optional)
            If specified, then use this minimum frequency rather than one
            chosen based on the size of the baseline.
        maximum_frequency : float (optional)
            If specified, then use this maximum frequency rather than one
            chosen based on the average nyquist frequency.

        Returns
        -------
        freq : ndarray
            an array `freq_min + freq_spacing * np.arange(N_freq)`.
        """
        t = np.asanyarray(t)
        baseline = t.max() - t.min()
        n_samples = t.size

        df = 1. / baseline / samples_per_peak

        if minimum_frequency is not None:
            f0 = minimum_frequency
        else:
            f0 = 0.5 * df

        if maximum_frequency is not None:
            Nf = int(np.ceil((maximum_frequency - f0) / df))
        else:
            Nf = int(0.5 * samples_per_peak * nyquist_factor * n_samples)

        return f0 + df * np.arange(Nf)
