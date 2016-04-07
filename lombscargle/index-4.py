import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from lombscargle import LombScargle

rand = np.random.RandomState(42)
t = 100 * rand.rand(100)
dy = 0.1
y = np.sin(2 * np.pi * t) + dy * rand.randn(100)

frequency, power = LombScargle(t, y, dy).autopower(minimum_frequency=0.1,
                                                   maximum_frequency=1.9)
best_frequency = frequency[np.argmax(power)]
phase_fit = np.linspace(0, 1)
y_fit = LombScargle(t, y, dy).model(t=phase_fit / best_frequency,
                                    frequency=best_frequency)
phase = (t * best_frequency) % 1

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.errorbar(phase, y, dy, fmt='o', mew=0, capsize=0, elinewidth=1.5)
ax.plot(phase_fit, y_fit, color='black')
ax.invert_yaxis()
ax.set(xlabel='phase',
       ylabel='magnitude',
       title='phased data at frequency={0:.2f}'.format(best_frequency))