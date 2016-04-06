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

Basic Usage
===========
The simplest usage of the Lomb-Scargle tool is via the :class:`LombScargle`
class. For example, consider the following data::

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

The periodogram also can handle heteroscedastic errors. For example:

>>> t = 100 * rand.rand(100)
>>> dy = rand.rand(100)
>>> y = np.sin(2 * np.pi * t) + dy * rand.randn(100)
>>> frequency, power = LombScargle(t, y, dy).autopower()

An example of computing the periodogram for a more realistic dataset is
shown in the following figure. This is 50 nightly observations of an
RR Lyrae-like variable star, with lightcurve shape that is more complicated
than a simple sine wave:

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt


    def simulated_data(N, rseed=0, period=0.6):
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
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    fig.subplots_adjust(bottom=0.12)  # stop labels from getting cut-off

    t, mag, dmag = simulated_data(50)

    ax[0].errorbar(t, mag, dmag, fmt='ok', elinewidth=1.5, capsize=0)
    ax[0].invert_yaxis()
    ax[0].set(xlabel='time (days)',
              ylabel='Observed Magnitude')

    freq, PLS = LombScargle(t, mag, dmag).autopower()

    ax[1].plot(1. / freq, PLS)
    ax[1].set(xlabel='period (days)',
              ylabel='Lomb-Scargle Power',
              xlim=(0.4, 1.0));

    fig.suptitle('Lomb-Scargle Periodogram (period=0.6 days)');

References
==========
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
