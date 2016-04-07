.. _lombscargle:

*******************************************
Lomb-Scargle Periodograms (``lombscargle``)
*******************************************

The Lomb-Scargle Periodogram (after Lomb [1]_, and Scargle [2]_)
is a commonly-used statistical tool designed to detect periodic signals
in unevenly-spaced observations.
The ``lombscargle`` package contains a unified interface to several
implementations of the Lomb-Scargle periodogram, including a fast *O[NlogN]*
implementation following the algorithm presented by Press & Rybicki [3]_.

Quick Start Guide
=================
The simplest usage of the Lomb-Scargle tool is via the
:class:`~lombscargle.LombScargle` class.
For example, consider the following data::

    >>> import numpy as np
    >>> rand = np.random.RandomState(42)
    >>> t = 100 * rand.rand(100)
    >>> y = np.sin(2 * np.pi * t) + 0.1 * rand.randn(100)

These are 100 noisy measurements taken at irregular times, with a frequency
of 1. The Lomb-Scargle periodogram, with frequency automatically chosen based
on the input data, can be computed as follows::

   >>> from lombscargle import LombScargle
   >>> frequency, power = LombScargle(t, y).autopower()

Plotting the result with matplotlib gives::

   >>> import matplotlib.pyplot as plt
   >>> plt.plot(frequency, power)   # doctest: +SKIP

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    rand = np.random.RandomState(42)
    t = 100 * rand.rand(100)
    y = np.sin(2 * np.pi * t) + 0.1 * rand.randn(100)

    from lombscargle import LombScargle
    frequency, power = LombScargle(t, y).autopower()
    fig = plt.figure(figsize=(6, 4.5))
    plt.plot(frequency, power)

The periodogram shows a clear spike at a frequency of 1, as we would expect
from the data we constructed.

Measurement Uncertainties
-------------------------

The periodogram also can handle data with measurement uncertainties.
For example, if all uncertainties are the same, you can pass a scalar:

>>> dy = 0.1
>>> frequency, power = LombScargle(t, y, dy).autopower()

If uncertainties vary from observation to observation, you can pass them as
an array:

>>> dy = 0.1 * (1 + rand.rand(100))
>>> y = np.sin(2 * np.pi * t) + dy * rand.randn(100)
>>> frequency, power = LombScargle(t, y, dy).autopower()

Gaussian uncertainties are assumed, and ``dy`` here specifies the standard
deviation (not the variance).

Data and Periodogram Units
--------------------------
The code supports :class:`~astropy.units.Quantity` objects with units attached,
and will validate the inputs to make sure units are appropriate. For example:

>>> import astropy.units as u
>>> t_days = t * u.day
>>> y_mags = y * u.mag
>>> dy_mags = y * u.mag
>>> frequency, power = LombScargle(t_days, y_mags, dy_mags).autopower()
>>> frequency.unit
Unit("1 / d")
>>> power.unit
Unit(dimensionless)

Note that in the standard normalization, regardless of the units of the input,
the Lomb-Scargle power *P* is a dimensionless quantity satisfying *0 ≤ P ≤ 1*.


Specifying the Frequency
------------------------
With the ``autopower()`` method used above, a heuristic is applied to select
a suitable frequency grid. By default, the heuristic assumes that the width of
peaks is inversely proportional to the observation baseline, and that the
maximum frequency is a factor of 5 larger than the so-called "average Nyquist
frequency", computed based on the average observation spacing.

This heuristic is not universally useful, as the frequencies probed by
irregularly-sampled data can be much higher than the average Nyquist frequency.
For this reason, the heuristic can be tuned through keywords passed to the
:func:`~lombscargle.heuristics.baseline_heuristic` function. For example:

>>> frequency, power = LombScargle(t, y, dy).autopower(nyquist_factor=2)
>>> len(frequency), frequency.min(), frequency.max()
(500, 0.0010189890448009111, 1.0179700557561102)

Here the highest frequency is two times the average Nyquist frequency.
If we increase the ``nyquist_factor``, we can probe higher frequencies:

>>> frequency, power = LombScargle(t, y, dy).autopower(nyquist_factor=10)
>>> len(frequency), frequency.min(), frequency.max()
(2500, 0.0010189890448009111, 5.0939262349597545)

Alternatively, we can use the ``power()`` method to evaluate the periodogram
at a user-specified set of frequencies:

>>> frequency = np.linspace(0.5, 1.5, 1000)
>>> power = LombScargle(t, y, dy).power(freqency)

Note that the fastest Lomb-Scargle implementation requires regularly-spaced
frequencies; if frequencies are irregularly-spaced, a slower method will be
used instead.

Frequency Grid Spacing
^^^^^^^^^^^^^^^^^^^^^^

One common issue with user-specified frequencies is choosing too coarse a
grid, such that significant peaks lie between grid points and are missed
entirely.

For example, imagine you chose to evaluate your periodogram at 100 points:

>>> frequency = np.linspace(0.1, 1.9, 100)
>>> power = LombScargle(t, y, dy).power(frequency)
>>> plt.plot(frequency, power)   # doctest: +SKIP

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from lombscargle import LombScargle

    rand = np.random.RandomState(42)
    t = 100 * rand.rand(100)
    dy = 0.1
    y = np.sin(2 * np.pi * t) + dy * rand.randn(100)

    frequency = np.linspace(0.1, 1.9, 100)
    power = LombScargle(t, y, dy).power(frequency)

    plt.style.use('ggplot')
    plt.figure(figsize=(6, 4.5))
    plt.plot(frequency, power)
    plt.xlabel('frequency')
    plt.ylabel('Lomb-Scargle Power')
    plt.ylim(0, 1)

From this plot alone, one might conclude that no clear periodic signal exists
in the data.
But this conclusion is in error: there is in fact a strong periodic signal,
but the periodogram peak falls in the gap between your grid points!

A safer approach is to use the frequency heuristic to decide on the appropriate
grid spacing to use, optionally passing a minimum and maximum frequency to
the ``autopower`` method:

>>> frequency, power = LombScargle(t, y, dy).autopower(minimum_frequency=0.1,
...                                                    maximum_frequency=1.9)
>>> len(frequency)
884
>>> plt.plot(frequency, power)   # doctest: +SKIP

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from lombscargle import LombScargle

    rand = np.random.RandomState(42)
    t = 100 * rand.rand(100)
    dy = 0.1
    y = np.sin(2 * np.pi * t) + dy * rand.randn(100)

    frequency, power = LombScargle(t, y, dy).autopower(minimum_frequency=0.1,
                                                       maximum_frequency=1.9)

    plt.style.use('ggplot')
    plt.figure(figsize=(6, 4.5))
    plt.plot(frequency, power)
    plt.xlabel('frequency')
    plt.ylabel('Lomb-Scargle Power')
    plt.ylim(0, 1)

With a finer grid (here 884 points between 0.1 and 1.9),
it is clear that there is a very strong periodic signal in the data.

By default, the heuristic aims to have roughly five grid points across each
significant periodogram peak; this can be increased by changing the
``samples_per_peak`` argument:

>>> frequency, power = LombScargle(t, y, dy).autopower(minimum_frequency=0.1,
...                                                    maximum_frequency=1.9,
...                                                    samples_per_peak=10)
>>> len(frequency)
1767

Note that the width of the peak scales inversely with the baseline of the
observations (i.e. the difference between the maximum and minimum time), and
the required number of grid points will scale linearly with the size of the
baseline.

Example
=======

An example of computing the periodogram for a more realistic dataset is
shown in the following figure. The simulated data here consist of
50 nightly observations of an RR Lyrae-like variable star,
with lightcurve shape that is more complicated than a simple sine wave:

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt


    def simulated_data(N, rseed=0, period=0.4):
        t = np.linspace(0, 1, 1000)
        rand = np.random.RandomState(rseed)
        t = np.arange(N, dtype=float)
        t += 0.1 * rand.randn(N)
        dmag = 0.05 + 0.2 * rand.rand(N)

        omega = 2 * np.pi / period
        coeffs = np.array([-1, -0.4, -0.1])
        n = np.arange(1, len(coeffs) + 1)[:, None]
        mag = 17 + np.dot(coeffs, np.sin(n * omega * t)) + dmag * rand.randn(N)

        return t, mag, dmag

    from lombscargle import LombScargle

    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(left=0.07, right=0.95)
    fig.subplots_adjust(bottom=0.12)  # stop labels from getting cut-off

    t, mag, dmag = simulated_data(50)

    ax[0].errorbar(t, mag, dmag, fmt='ok', elinewidth=1.5, capsize=0)
    ax[0].invert_yaxis()
    ax[0].set(xlabel='time (days)',
              ylabel='Observed Magnitude')

    freq, PLS = LombScargle(t, mag, dmag).autopower(minimum_frequency=1 / 1.2,
                                                    maximum_frequency=1 / 0.2)

    ax[1].plot(1. / freq, PLS)
    ax[1].set(xlabel='period (days)',
              ylabel='Lomb-Scargle Power',
              xlim=(0.2, 1.2),
              ylim=(0, 1));

    phase = (t * freq[np.argmax(PLS)] + 0.2) % 1
    inset = fig.add_axes([0.78, 0.56, 0.15, 0.3])
    inset.errorbar(phase, mag, dmag, fmt='.k', capsize=0)
    inset.invert_yaxis()
    inset.set_xlabel('phase')
    inset.set_ylabel('mag')

    fig.suptitle('Lomb-Scargle Periodogram (period=0.4 days)');

This example demonstrates that for irregularly-sampled data,
the Lomb-Scargle periodogram can be sensitive to frequencies higher
than the average Nyquist frequency: the above data are sampled at
an average rate of roughly one per day, and the periodogram
relatively cleanly reveals the true period of 0.4 days.


Literature References
=====================
.. [1] Lomb, N.R. "Least-squares frequency analysis of unequally spaced data".
       Ap&SS 39 pp. 447-462 (1976)
.. [2] Scargle, J. D. "Studies in astronomical time series analysis. II -
       Statistical aspects of spectral analysis of unevenly spaced data".
       ApJ 1:263 pp. 835-853 (1982)
.. [3] Press W.H. and Rybicki, G.B, "Fast algorithm for spectral analysis
       of unevenly sampled data". ApJ 1:338, p. 277 (1989)


API Reference
=============

.. automodapi:: lombscargle

.. automodapi:: lombscargle.implementations

.. automodapi:: lombscargle.heuristics
