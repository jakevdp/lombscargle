
.. _lombscargle-example:


Example
=======

An example of computing the periodogram for a more realistic dataset is
shown in the following figure. The simulated data here consist of
50 nightly observations of a vaguely RR Lyrae-like variable star,
with lightcurve shape that is more complicated than a simple sine wave:

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    from lombscargle import LombScargle


    def simulated_data(N, rseed=0, period=0.4, phase=0.1):
        t = np.linspace(0, 1, 1000)
        rand = np.random.RandomState(rseed)
        t = phase + np.arange(N, dtype=float)
        t += 0.1 * rand.randn(N)
        dmag = 0.02 + 0.05 * rand.rand(N)

        omega = 2 * np.pi / period
        coeffs = np.array([-1, -0.4, -0.1])
        n = np.arange(1, len(coeffs) + 1)[:, None]
        mag = (7.6 +  0.4 * np.dot(coeffs, np.sin(n * omega * t))
               + dmag * rand.randn(N))

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
    ax[0].set(xlabel='time (days)',
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
relatively cleanly reveals the true period of 0.4 days, though there
are aliases due to the interaction between the 0.4-day signal and the
roughly 1.0-day observing window.
