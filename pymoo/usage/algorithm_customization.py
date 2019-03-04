from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.optimize import minimize
from pymoo.util import plotting
from pymop.factory import get_problem

# create the optimization problem
problem = get_problem("zdt1")
pf = problem.pareto_front()

# --------------------------------------------------------------------------------
# Here we define our evolutionary operators to be used by the algorithm
# --------------------------------------------------------------------------------

sampling = RandomSampling()
crossover = SimulatedBinaryCrossover(prob_cross=1.0, eta_cross=5)
mutation = PolynomialMutation(prob_mut=0.1, eta_mut=10)


# then provide the operators to the minimize method
res = minimize(problem,
               method='nsga2',
               method_args={
                   'pop_size': 100,
                   'sampling': sampling,
                   'crossover': crossover,
                   'mutation': mutation
               },
               termination=('n_gen', 200),
               pf=pf,
               save_history=False,
               disp=True)
plotting.plot(pf, res.F, labels=["Pareto-front", "F"])
