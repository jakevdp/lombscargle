"""
Main Lomb-Scargle Implementation

The ``lombscargle`` function here is essentially a sophisticated switch
statement for the various implementations available in this submodule
"""
import numpy as np
from astropy import units

from .slow_impl import lombscargle_slow
from .fast_impl import lombscargle_fast
from .scipy_impl import lombscargle_scipy
from .matrix_impl import lombscargle_matrix


METHODS = {'slow': lombscargle_slow,
           'fast': lombscargle_fast,
           'matrix': lombscargle_matrix,
           'scipy': lombscargle_scipy}


def _validate_inputs(t, y, dy=None, frequency=None, strip_units=True):
    """Validation of input shapes & units

    This utility function serves a few purposes:

    - it validates that the shapes of t, y, and dy match, and broadcasts
      them to a common 1D shape
    - if any of t, y, day, or frequency are astropy Quantities (i.e. have
      units attached), it validates that the units are compatible, and does
      any necessary unit conversions
    - if ``strip_units == True``, it strips units from all the arrays
      before returning them.
    - all relevant units are returned in ``unit_dict``

    Parameters
    ----------
    t, y : array_like or Quantity
    dy, frequency : array_like or Quantity (optional)
    strip_units : bool (optional, default=True)
        if True, the returned quantities will have units stripped.

    Returns
    -------
    t, y, dy, frequency : ndarray, Quantity, or None
        reshaped and/or unit-stripped arrays
    unit_dict : dict
        dictionary of relevant units
    """
    if dy is None:
        t, y = np.broadcast_arrays(t, y, subok=True)
    else:
        t, y, dy = np.broadcast_arrays(t, y, dy, subok=True)

    if t.ndim != 1:
        raise ValueError("Input times & data must be one-dimensional")

    has_units = any(isinstance(arr, units.Quantity)
                    for arr in (t, y, dy, frequency))

    if has_units:
        power_unit = units.dimensionless_unscaled

        t = units.Quantity(t)
        y = units.Quantity(y)

        if frequency is not None:
            frequency = units.Quantity(frequency)
            if not t.unit.is_equivalent(1. / frequency.unit):
                raise ValueError("Units of frequency not equivalent to "
                                 "units of 1/t")
            t = units.Quantity(t, unit=1. / frequency.unit)

        if dy is not None:
            dy = units.Quantity(dy)
            if not y.unit.is_equivalent(dy.unit):
                raise ValueError("Units of y not equivalent to units of dy")
            dy = units.Quantity(dy, unit=y.unit)
    else:
        power_unit = 1

        t = np.asarray(t)
        y = np.asarray(y)
        if dy is not None:
            dy = np.asarray(dy)

    def get_unit(val):
        if isinstance(val, units.Quantity):
            return val.unit
        else:
            return 1

    unit_dict = {'t': get_unit(t),
                 'y': get_unit(y),
                 'dy': get_unit(y),
                 'frequency': 1. / get_unit(t),
                 'power': power_unit}

    def unit_strip(arr):
        if arr is None:
            return arr
        else:
            return np.asarray(arr)

    if strip_units:
        t, y, dy, frequency = map(unit_strip, (t, y, dy, frequency))

    return t, y, dy, frequency, unit_dict


def _get_frequency_grid(frequency, assume_regular_frequency=False):
    """Utility to get grid parameters from a frequency array

    Parameters
    ----------
    frequency : array_like or Quantity
        input frequency grid
    assume_regular_frequency : bool (default = False)
        if True, then do not check whether frequency is a regular grid

    Returns
    -------
    f0, df, N : scalars
        Parameters such that all(frequency == f0 + df * np.arange(N))
    """
    frequency = np.asarray(frequency)
    if frequency.ndim != 1:
        raise ValueError("frequency grid must be 1 dimensional")
    elif len(frequency) == 1:
        return frequency[0], frequency[0], 1
    elif not assume_regular_frequency:
        diff = frequency[1:] - frequency[:-1]
        if not np.allclose(diff[0], diff):
            raise ValueError("frequency must be a regular grid")

    return frequency[0], frequency[1] - frequency[0], len(frequency)


def lombscargle(t, y, dy=None,
                frequency=None,
                method='auto',
                assume_regular_frequency=False,
                normalization='normalized',
                fit_bias=True, center_data=True,
                method_kwds=None):
    """
    Compute the Lomb-scargle Periodogram

    Parameters
    ----------
    t : array_like
        sequence of observation times
    y : array_like
        sequence of observations associated with times t
    dy : float or array_like (optional)
        error or sequence of observational errors associated with times t
    frequency : array_like
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
        - `slow`: use the O[N^2] pure-python implementation
        - `matrix`: use the O[N^2] matrix/linear-fitting implementation
        - `scipy`: use ``scipy.signal.lombscargle``, which is an O[N^2]
          implementation written in C. Note that this does not support
          heteroskedastic errors.

    assume_regular_frequency : bool (optional)
        if True, assume that the input frequency is of the form
        freq = f0 + df * np.arange(N). Only referenced if method is 'auto'
        or 'fast'.
    normalization : string (optional, default='normalized')
        Normalization to use for the periodogram. Options are 'normalized' or
        'unnormalized'.
    fit_bias : bool (optional, default=True)
        if True, include a constant offet as part of the model at each
        frequency. This can lead to more accurate results, especially in then
        case of incomplete phase coverage.
    center_data : bool (optional, default=True)
        if True, pre-center the data by subtracting the weighted mean
        of the input data. This is especially important if `fit_bias = False`
    method_kwds : dict (optional)
        additional keywords to pass to the lomb-scargle method

    Returns
    -------
    PLS : array_like
        Lomb-Scargle power associated with each frequency omega
    """
    if frequency is None:
        raise ValueError("Must supply a valid frequency. If you would like "
                         "an automatic frequency grid, use the "
                         "LombScargle.autopower() method.")

    t, y, dy, frequency, unit_dict = _validate_inputs(t, y, dy, frequency)

    output_shape = frequency.shape
    frequency = frequency.ravel()

    if method == 'auto':
        # TODO: make more careful choices here
        if len(frequency) > 100:
            method = 'fast'
        elif dy is None and not fit_bias:
            method = 'scipy'
        else:
            method = 'slow'

    if method == 'fast':
        f0, df, Nf = _get_frequency_grid(frequency, assume_regular_frequency)
        PLS = lombscargle_fast(t, y, dy=dy, f0=f0, df=df, Nf=Nf,
                               center_data=center_data,
                               fit_bias=fit_bias,
                               normalization=normalization,
                               **(method_kwds or {}))
    elif method == 'scipy':
        assert not fit_bias
        PLS = lombscargle_scipy(t, y, dy=dy, frequency=frequency,
                                center_data=center_data,
                                normalization=normalization,
                                **(method_kwds or {}))
    else:
        PLS = METHODS[method](t, y, dy=dy, frequency=frequency,
                              center_data=center_data,
                              fit_bias=fit_bias,
                              normalization=normalization,
                              **(method_kwds or {}))

    return PLS.reshape(output_shape) * unit_dict['power']
