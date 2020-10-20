import numpy as np
from pymoo.factory import get_performance_indicator

hv = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2]))
A = np.array([[0.5,0.25], [0.25, 0.5]])
print("hv", hv.calc(A))