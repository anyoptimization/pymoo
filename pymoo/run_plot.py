
import numpy as np
import matplotlib.pyplot as plt

f = np.loadtxt('../../results/pynsga_ZDT6_02.out')

plt.scatter(f[:, 0], f[:, 1])
plt.show()

print f