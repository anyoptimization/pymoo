import numpy as np
import sys
sys.path.append('/Users/yash/Desktop/research/moo/pymoo/pymoo/cpp/hypervolume/build')
import hypervolume
A = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
B = np.array([2, 2, 2, 2])

v = hypervolume.calculate(4, 2, A, B)
print(v)