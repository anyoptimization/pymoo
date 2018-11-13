

# create the optimization problem
from pymoo.model.termination import MaximumGenerationTermination

from pymoo.optimize import minimize

from pymop.factory import get_problem

problem = get_problem("zdt1")
pf = problem.pareto_front()


# ----------------------------------------------
# Number of Generations
# ----------------------------------------------
res = minimize(problem,
               method='nsga2',
               method_args={'pop_size': 20},
               termination=('n_gen', 5),
               pf=pf,
               disp=True)


# ----------------------------------------------
# Number of Function Evaluations
# ----------------------------------------------

res = minimize(problem,
               method='nsga2',
               method_args={'pop_size': 20},
               termination=('n_eval', 500),
               pf=pf,
               disp=True)

# ----------------------------------------------
# Until specific IGD is reached
# ----------------------------------------------

res = minimize(problem,
               method='nsga2',
               method_args={'pop_size': 100},
               termination=('igd', 0.1),
               pf=pf,
               disp=True)


# ----------------------------------------------
# Using a Custom Object - (Custom Termination Criteria can be defined)
# ----------------------------------------------

res = minimize(problem,
               method='nsga2',
               method_args={'pop_size': 40},
               termination=MaximumGenerationTermination(25),
               pf=pf,
               disp=True)




