from abc import abstractmethod

from pymoo.model import random
from pymoo.model.evaluator import Evaluator
from pymoo.util.misc import calc_constraint_violation
from pymoo.util.non_dominated_rank import NonDominatedRank


class Algorithm:
    """

    This class represents the abstract class for any algorithm to be implemented. Most importantly it
    provides the solve method that is used to optimize a given problem.

    """

    def solve(self, problem, evaluator, seed=1, return_only_feasible=True, return_only_non_dominated=True):
        """

        Solve a given problem by a given evaluator. The evaluator determines the termination condition and
        can either have a maximum budget, hypervolume or whatever. The problem can be any problem the algorithm
        is able to solve.

        Parameters
        ----------

        problem: class
            Problem to be solved by the algorithm

        evaluator: class
            object that evaluates and saves the number of evaluations and determines the stopping condition

        seed: int
            Random seed for this run. Before the algorithm starts this seed is set.

        return_only_feasible:
            If true, only feasible solutions are returned.

        return_only_non_dominated
            If true, only the non dominated solutions are returned. Otherwise, it might be - dependend on the
            algorithm - the final population

        Returns
        -------
        X: matrix
            Design space

        F: matrix
            Objective space

        G: matrix
            Constraint space

        """

        # set the random seed
        random.seed(seed)

        if not isinstance(evaluator, Evaluator):
            evaluator = Evaluator(evaluator)

        # call the algorithm to solve the problem
        X, F, G = self._solve(problem, evaluator)

        if return_only_feasible:
            if G is not None and G.shape[0] == len(F) and G.shape[1] > 0:
                cv = calc_constraint_violation(G)
                X = X[cv, :]
                F = F[cv, :]
                if G is not None:
                    G = G[cv, :]

        if return_only_non_dominated:
            idx_non_dom = NonDominatedRank.calc_as_fronts(F,G)[0]
            X = X[idx_non_dom, :]
            F = F[idx_non_dom, :]
            if G is not None:
                G = G[idx_non_dom, :]

        return X, F, G

    @abstractmethod
    def _solve(self, problem, evaluator):
        pass
