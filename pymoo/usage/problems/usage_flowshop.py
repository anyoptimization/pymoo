from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.operators.crossover.order_crossover import OrderCrossover
from pymoo.operators.mutation.inversion_mutation import InversionMutation
from pymoo.operators.sampling.random_permutation_sampling import RandomPermutationSampling
from pymoo.optimize import minimize
from pymoo.problems.single.flowshop_scheduling import visualize, create_random_flowshop_problem

problem = create_random_flowshop_problem(n_machines=5, n_jobs=10, seed=1)

# solve the problem using GA
algorithm = GA(
    pop_size=200,
    eliminate_duplicates=True,
    sampling=RandomPermutationSampling(),
    mutation=InversionMutation(),
    crossover=OrderCrossover()
)
res = minimize(
    problem,
    algorithm,
    seed=2,
    verbose=False
)

visualize(problem, res.X)
