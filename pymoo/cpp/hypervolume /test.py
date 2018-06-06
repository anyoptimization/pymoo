import numpy as np
# import cppimport
from build import example


A = np.array([[1, 0, 0, 1],
              [0, 1, 1, 0]])
B = np.array([2, 2, 2, 2])
v = example.calculate(4, 2, A, B)
print(v)