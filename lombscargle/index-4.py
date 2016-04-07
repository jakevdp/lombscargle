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