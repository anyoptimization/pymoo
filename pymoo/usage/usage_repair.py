import numpy as np

from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.model.repair import Repair
from pymoo.optimize import minimize
from pymop import create_random_knapsack_problem


class ConsiderMaximumWeightRepair(Repair):

    def _do(self, problem, pop, **kwargs):

        # maximum capacity for the problem
        Q = problem.C

        # the packing plan for the whole population (each row one individual)
        Z = pop.get("X")

        # the corresponding weight of each individual
        weights = (Z * problem.W).sum(axis=1)

        # now repair each indvidiual i
        for i in range(len(Z)):

            # the packing plan for i
            z = Z[i]

            # while the maximum capacity violation holds
            while weights[i] > Q:
                # randomly select an item currently picked
                item_to_remove = np.random.choice(np.where(z)[0])

                # and remove it
                z[item_to_remove] = False

                # adjust the weight
                weights[i] -= problem.W[item_to_remove]

        # set the design variables for the population
        pop.set("X", Z)
        return pop


method = get_algorithm("ga",
                       pop_size=200,
                       sampling=get_sampling("bin_random"),
                       crossover=get_crossover("bin_hux"),
                       mutation=get_mutation("bin_bitflip"),
                       repair=ConsiderMaximumWeightRepair(),
                       elimate_duplicates=True)

res = minimize(create_random_knapsack_problem(30),
               method,
               termination=('n_gen', 10),
               disp=True)
