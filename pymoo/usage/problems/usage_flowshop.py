from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.operators.crossover.order_crossover import OrderCrossover
from pymoo.operators.mutation.inversion_mutation import InversionMutation
from pymoo.operators.sampling.random_permutation_sampling import PermutationRandomSampling
from pymoo.optimize import minimize
from pymoo.problems.single.flowshop_scheduling import visualize, create_random_flowshop_problem
from pymoo.util.termination.default import SingleObjectiveDefaultTermination


problem = create_random_flowshop_problem(n_machines=5, n_jobs=10, seed=1)

algorithm = GA(
    pop_size=20,
    eliminate_duplicates=True,
    sampling=PermutationRandomSampling(),
    mutation=InversionMutation(),
    crossover=OrderCrossover()
)

# if the algorithm did not improve the last 200 generations then it will terminate
termination = SingleObjectiveDefaultTermination(n_last=50, n_max_gen=10000)


res = minimize(
    problem,
    algorithm,
    termination,
    seed=1,
    verbose=True
)

visualize(problem, res.X)
