import random

from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.model.mating import Mating
from pymoo.model.repair import NoRepair
from pymoo.operators.crossover.differental_evolution_crossover import DifferentialEvolutionCrossover
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.no_mutation import NoMutation
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.selection.random_selection import RandomSelection
from pymoo.optimize import minimize
from pymoo.problems.single import Rastrigin


class EnsembleMating(Mating):

    def __init__(self, selections, crossovers, mutations, repairs, **kwargs) -> None:
        super().__init__(None, None, None, **kwargs)
        self.selections = selections
        self.crossovers = crossovers
        self.mutations = mutations
        self.repairs = repairs if len(repairs) > 0 else [NoRepair()]

    def do(self, problem, pop, n_offsprings, **kwargs):
        # randomly choose a combination to be tried
        self.selection = random.choice(self.selections)
        self.crossover = random.choice(self.crossovers)
        self.mutation = random.choice(self.mutations)
        self.repair = random.choice(self.repairs)

        off = super().do(problem, pop, n_offsprings, **kwargs)
        return off


selections = [RandomSelection()]

# define all the crossovers to be tried
crossovers = [SimulatedBinaryCrossover(10.0), SimulatedBinaryCrossover(30.0), DifferentialEvolutionCrossover()]
# COMMENT out this line to only use the SBX crossover with one eta value
# crossovers = [SimulatedBinaryCrossover(30)]

mutations = [NoMutation(), PolynomialMutation(10.0), PolynomialMutation(30.0)]
repairs = []

ensemble = EnsembleMating(selections, crossovers, mutations, repairs)

problem = Rastrigin(n_var=30)

algorithm = GA(
    pop_size=100,
    mating=ensemble,
    eliminate_duplicates=True)

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
