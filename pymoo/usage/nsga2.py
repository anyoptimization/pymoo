
from pymoo.factory import get_algorithm
from pymoo.optimize import minimize
from pymoo.util import plotting
from pymop.factory import get_problem

# create the algorithm object
method = get_algorithm("nsga2",
                      pop_size=100,
                      elimate_duplicates=True)

# create the test problem
problem = get_problem("zdt1")
pf = problem.pareto_front()

# execute the optimization
res = minimize(problem,
               method,
               termination=('n_gen', 200),
               pf=pf,
               disp=False)

plotting.plot(pf, res.F, labels=["Pareto-front", "F"])
