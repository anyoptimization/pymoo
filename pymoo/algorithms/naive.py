import copy

from scipy.stats import stats

from metamodels.gpy_metamodel import GPyMetaModel
from metamodels.sklearn_metamodel_dace import SKLearnDACEMetaModel
from model.algorithm import Algorithm
from operators.lhs_factory import LHS
from rand.default_random_generator import DefaultRandomGenerator
from vendor.pygmo_solver import solve_by_moad


class NaiveMetamodelAlgorithm(Algorithm):
    def __init__(self, n_iter):
        self.n_iter = n_iter

    def solve_(self, problem, evaluator,rnd=DefaultRandomGenerator()):

        # initial sampling
        X = LHS().sample(int(evaluator.n_eval - self.n_iter), problem.xl, problem.xu)
        F, _ = evaluator.eval(problem,X)

        # learn a metamodel on the data
        meta_model = GPyMetaModel()
        meta_model.fit(X,F)

        # optimize the problem on the metamodel
        meta_model_problem = copy.deepcopy(problem)

        # evaluation function for the algorithm to be optimized on the metamodel
        def evaluate_(x, f):
            f[:, :], std = meta_model.predict(x)

        meta_model_problem.func = evaluate_

        # optimize on the metamodel
        x_hat, f_hat, _ = solve_by_moad(meta_model_problem, pop_size=self.n_iter)

        f_hat_true,_ = evaluator.eval(problem, x_hat)

        return x_hat, f_hat_true, None