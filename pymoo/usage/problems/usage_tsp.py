from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.operators.crossover.order_crossover import OrderCrossover
from pymoo.operators.mutation.inversion_mutation import InversionMutation
from pymoo.operators.sampling.random_permutation_sampling import RandomPermutationSampling
from pymoo.optimize import minimize
from pymoo.problems.single.traveling_salesman import visualize, create_random_tsp_problem


problem = create_random_tsp_problem(10, 100, seed=1)

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
    seed=1,
    verbose=False
)

print(res.F)
print(res.algorithm.evaluator.n_eval)
visualize(problem, res.X)

