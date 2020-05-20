import numpy as np
import matplotlib.pyplot as plt

from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.optimize import minimize
from pymoo.problems.single.flowshop_scheduling import FlowshopScheduling
from pymoo.operators.crossover.order_crossover import OrderCrossover
from pymoo.operators.mutation.inversion_mutation import InversionMutation
from pymoo.operators.sampling.random_permutation_sampling import RandomPermutationSampling


# visualize a solution
def visualize(problem, x, path=None, label=True):
    with plt.style.context('ggplot'):
        n_machines, n_jobs = problem.data.shape
        machine_times = problem.get_machine_times(x)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        Y = np.flip(np.arange(n_machines))

        for i in range(n_machines):
            for j in range(n_jobs):
                width = problem.data[i][x[j]]
                left = machine_times[i][j]
                ax.barh(Y[i], width, left=left,
                        align='center', color='gray',
                        edgecolor='black', linewidth=0.8
                        )
                if label:
                    ax.text((left + width / 2), Y[i], str(x[j] + 1), ha='center', va='center', color='white',
                            fontsize=15)
        ax.set_xlabel("Time")
        ax.set_yticks(np.arange(n_machines))
        ax.set_yticklabels(["M%d" % (i + 1) for i in Y])
        ax.set_title("Makespan: {}".format(problem.makespan(x)))
        if path is not None:
            plt.savefig(path)
        plt.show()


# create random problem
num_machines = 5
num_jobs = 10
processing_times = np.random.random((num_machines, num_jobs))*50+50

problem = FlowshopScheduling(processing_times)

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
    termination=('n_eval', 50000),
    verbose=False
)
visualize(problem, res.X)