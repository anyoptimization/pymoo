from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.crossover.bin_uniform_crossover import BinaryUniformCrossover
from pymoo.operators.crossover.real_simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.bin_bitflip_mutation import BinaryBitflipMutation
from pymoo.operators.mutation.real_polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.bin_random_sampling import BinaryRandomSampling
from pymoo.operators.sampling.real_random_sampling import RealRandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.operators.survival.fitness_survival import FitnessSurvival


class SingleObjectiveGeneticAlgorithm(GeneticAlgorithm):
    def __init__(self, var_type, pop_size=100, verbose=1):

        if var_type == "real":
            super().__init__(
                pop_size=pop_size,
                sampling=RealRandomSampling(),
                selection=TournamentSelection(),
                crossover=SimulatedBinaryCrossover(),
                mutation=PolynomialMutation(),
                survival=FitnessSurvival(),
                verbose=verbose
            )
        elif var_type == "binary":
            super().__init__(
                pop_size=pop_size,
                sampling=BinaryRandomSampling(),
                selection=TournamentSelection(),
                crossover=BinaryUniformCrossover(),
                mutation=BinaryBitflipMutation(),
                survival=FitnessSurvival(),
                verbose=verbose,
                eliminate_duplicates=True
            )


