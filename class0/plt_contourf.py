import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

n = 100
x = np.linspace(-10, 10, n)
y = np.linspace(-10, 10, n)
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(8, 6))
Z = X + Y
plt.subplot(221)
plt.pcolormesh(X, Y, Z, cmap='rainbow')
plt.subplot(222)
plt.contourf(X, Y, Z, 100, cmap='rainbow')

Z = X**2 + Y**2
plt.subplot(223)
plt.contour(X, Y, Z, 20, cmap='rainbow')
plt.subplot(224)
plt.contourf(X, Y, Z, 20, cmap='rainbow')

plt.tight_layout()
plt.show()
