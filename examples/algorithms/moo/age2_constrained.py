from pymoo.indicators.igd import IGD
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.optimize import minimize

from pymoo.problems.many import C1DTLZ1, DC1DTLZ1, DC1DTLZ3, DC2DTLZ1, DC2DTLZ3, DC3DTLZ1, DC3DTLZ3, C1DTLZ3, \
    C2DTLZ2, C3DTLZ1, C3DTLZ4
import ray
import numpy as np

benchmark_algorithms = [
    AGEMOEA2(),
]

benchmark_problems = [
    C1DTLZ1, DC1DTLZ1, DC1DTLZ3, DC2DTLZ1, DC2DTLZ3, DC3DTLZ1, DC3DTLZ3, C1DTLZ3, C2DTLZ2, C3DTLZ1, C3DTLZ4
]


def run_benchmark(problem_class, algorithm):
    # Instantiate the problem
    problem = problem_class()

    res = minimize(
        problem,
        algorithm,
        pop_size=100,
        verbose=True,
        seed=1,
        termination=('n_gen', 2000)
    )

    # Step 4: Generate the reference points
    ref_dirs = get_reference_directions("uniform", problem.n_obj, n_points=528)

    # Obtain the true Pareto front (for synthetic problems)
    pareto_front = problem.pareto_front(ref_dirs)

    # Calculate IGD
    if res.F is None:
        igd = np.Infinity
    else:
        igd = IGD(pareto_front)(res.F)

    result = {
        "problem": problem,
        "algorithm": algorithm,
        "result": res,
        "igd": igd
    }

    return result


tasks = []
for problem in benchmark_problems:
    for algorithm in benchmark_algorithms:
        tasks.append(ray.remote(run_benchmark).remote(problem, algorithm))
result = ray.get(tasks)

for res in result:
    print(f"Algorithm = {res['algorithm'].__class__.__name__}, "
          f"Problem = {res['problem'].__class__.__name__}, "
          f"IGD = {res['igd']}")
