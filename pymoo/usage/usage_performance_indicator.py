# START load_data
import numpy as np
from pymoo.factory import get_problem
from pymoo.visualization.scatter import Scatter

# The pareto front of a scaled zdt1 problem
pf = get_problem("zdt1").pareto_front()

# The result found by an algorithm
A = pf[::10] * 1.1

# plot the result
Scatter(legend=True).add(pf, label="Pareto-front").add(A, label="Result").show()
# END load_data





# START gd
from pymoo.factory import get_performance_indicator

gd = get_performance_indicator("gd", pf)
print("GD", gd.calc(A))
# END gd



# START gd_plus
from pymoo.factory import get_performance_indicator

gd_plus = get_performance_indicator("gd+", pf)
print("GD+", gd_plus.calc(A))
# END gd_plus


# START igd
from pymoo.factory import get_performance_indicator

igd = get_performance_indicator("igd", pf)
print("IGD", igd.calc(A))
# END igd



# START igd_plus
from pymoo.factory import get_performance_indicator

igd_plus = get_performance_indicator("igd+", pf)
print("IGD+", igd_plus.calc(A))
# END igd_plus


# START hv
from pymoo.factory import get_performance_indicator

hv = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2]))
print("hv", hv.calc(A))
# END hv