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