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