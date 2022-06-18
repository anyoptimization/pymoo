import numpy as np

from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.problems import get_problem

pf = get_problem("zdt1").pareto_front()
F = pf[::10] * 1.1

for clazz in [GD, GDPlus, IGD, IGDPlus]:
    print(clazz.__name__, clazz(pf)(F))

ref_point = np.array([1.2, 1.2])

ind = HV(ref_point=ref_point)
print("HV", ind(F))
