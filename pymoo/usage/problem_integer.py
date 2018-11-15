import numpy as np

from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.optimize import minimize
from pymop import Problem


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=2, n_obj=1, n_constr=1, xl=0, xu=10, type_var=np.int)

    def _evaluate(self, x, f, g, *args, **kwargs):
        f[:, 0] = - np.min(x * [3, 1], axis=1)
        g[:, 0] = x[:, 0] + x[:, 1] - 10


def repair(problem, pop, **kwargs):
    pop.set("X", np.round(pop.get("X")).astype(np.int))
    return pop


res = minimize(MyProblem(),
               method='ga',
               method_args={
                   'pop_size': 20,
                   'crossover': SimulatedBinaryCrossover(prob_cross=0.9, eta_cross=3),
                   'mutation': PolynomialMutation(eta_mut=2),
                   'eliminate_duplicates': True,
                   'func_repair': repair
               },
               termination=('n_gen', 30),
               disp=True)

print("Best solution found: %s" % res.X)
print("Function value: %s" % res.F)
print("Constraint violation: %s" % res.CV)
