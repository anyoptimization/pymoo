# create the optimization problem
import numpy as np

from pymoo.model.population import Population
from pymoo.optimize import minimize
from pymop.factory import get_problem

problem = get_problem("rastrigin")

pop_size = 100

pop = Population(pop_size)
pop.set("X", np.random.random((pop_size, problem.n_var)))


res = minimize(problem,
               method='ga',
               method_args={'pop_size': pop_size,
                            'sampling': pop
                            },
               termination=('n_gen', 200),
               disp=True)
