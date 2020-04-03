from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination
from pymoo.visualization.scatter import Scatter

problem = get_problem("welded_beam")

algorithm = NSGA2(pop_size=200,
                  crossover=SimulatedBinaryCrossover(eta=20, prob=0.9),
                  mutation=PolynomialMutation(prob=0.25, eta=40),
                  )

termination = MultiObjectiveSpaceToleranceTermination(n_last=60)

res = minimize(problem,
               algorithm,
               pf=False,
               seed=10,
               verbose=True)

print(res.algorithm.n_gen)
Scatter().add(res.F, s=20).add(problem.pareto_front(), plot_type="line", color="black").show()
