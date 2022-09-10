from scipy.interpolate import lagrange
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = 10 * np.random.rand(15)
y = x + 0.5 * np.random.rand(15)
p = lagrange(x[0:10], y[0:10])
xp = np.linspace(0, 10, 100)
yp = p(xp)
plt.scatter(x[0:10], y[0:10], c='k')
plt.scatter(x[10:15], y[10:15], c='b', marker='+')
plt.scatter(x[10:15], p(y[10:15]), c='r', marker='*')
plt.plot(xp, yp, 'k:')
plt.xlabel('x'), plt.ylabel('y')
plt.ylim((-300, 400))
plt.show()
