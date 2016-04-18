
.. _lombscargle-example:


Example
=======

An example of computing the periodogram for a more realistic dataset is
shown in the following figure. The simulated data here consist of
50 nightly observations of a simulated RR Lyrae-like variable star,
with lightcurve shape that is more complicated than a simple sine wave:

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    from lombscargle import LombScargle


    def simulated_data(N, rseed=2, period=0.41, phase=0.0):
        """Simulate data based from a pre-computed empirical fit"""

        # coefficients from a 5-term Fourier fit to SDSS object 1019544
        coeffs = [-0.0191, 0.1375, -0.1968, 0.0959, 0.075,
                  -0.0686, 0.0307, -0.0045, -0.0421, 0.0216, 0.0041]

        rand = np.random.RandomState(rseed)
        t = phase + np.arange(N, dtype=float)
        t += 0.1 * rand.randn(N)
        dmag = 0.01 + 0.03 * rand.rand(N)

        omega = 2 * np.pi / period
        n = np.arange(6)[:, None]

        mag = (15 + dmag * rand.randn(N)
               + np.dot(coeffs[::2], np.cos(n * omega * t)) +
               + np.dot(coeffs[1::2], np.sin(n[1:] * omega * t)))

        return t, mag, dmag


    # generate data and compute the periodogram
    t, mag, dmag = simulated_data(50)
    freq, PLS = LombScargle(t, mag, dmag).autopower(minimum_frequency=1 / 1.2,
                                                    maximum_frequency=1 / 0.2)
    best_freq = freq[np.argmax(PLS)]
    phase = (t * best_freq) % 1

    # compute the best-fit model
    phase_fit = np.linspace(0, 1)
    mag_fit = LombScargle(t, mag, dmag).model(t=phase_fit / best_freq,
                                              frequency=best_freq)

    # set up the figure & axes for plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Lomb-Scargle Periodogram (period=0.4 days)')
    fig.subplots_adjust(bottom=0.12, left=0.07, right=0.95)
    inset = fig.add_axes([0.78, 0.56, 0.15, 0.3])

    # plot the raw data
    ax[0].errorbar(t, mag, dmag, fmt='ok', elinewidth=1.5, capsize=0)
    ax[0].invert_yaxis()
    ax[0].set(xlim=(0, 50),
              xlabel='Observation time (days)',
              ylabel='Observed Magnitude')

    # plot the periodogram
    ax[1].plot(1. / freq, PLS)
    ax[1].set(xlabel='period (days)',
              ylabel='Lomb-Scargle Power',
              xlim=(0.2, 1.2),
              ylim=(0, 1));

    # plot the phased data & model in the inset
    inset.errorbar(phase, mag, dmag, fmt='.k', capsize=0)
    inset.plot(phase_fit, mag_fit)
    inset.invert_yaxis()
    inset.set_xlabel('phase')
    inset.set_ylabel('mag')


This example demonstrates that for irregularly-sampled data,
the Lomb-Scargle periodogram can be sensitive to frequencies higher
than the average Nyquist frequency: the above data are sampled at
an average rate of roughly one per day, and the periodogram
relatively cleanly reveals the true period of 0.41 days, though there
are aliases due to the interaction between the 0.41-day signal and the
roughly 1.0-day observing window.
