import random

import numpy as np

from model.algorithm import Algorithm
from model.individual import Individual
from model.problem import single_objective_problem_by_asf
from operators.polynomial_mutation import PolynomialMutation
from operators.random_factory import RandomFactory
from operators.simulated_binary_crossover import SimulatedBinaryCrossover
from rand.default_random_generator import DefaultRandomGenerator
from util.dominator import Dominator
from util.misc import evaluate
from util.rank_and_crowding import RankAndCrowdingSurvival


class DecomposeAlgorithm(Algorithm):
    def __init__(self,
                 weights=None # that should be used to decompose the problem -> if None it is done randomly
                 ):
        self.weights = weights

    def solve_(self, problem, evaluator, rnd=DefaultRandomGenerator()):

        single_objective_problem_by_asf()
        pass

