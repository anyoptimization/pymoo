import copy
from abc import abstractmethod

import numpy as np

from pymoo.model.result import Result
from pymoo.model.termination import MaximumFunctionCallTermination
from pymoo.rand import random
from pymoo.util.non_dominated_sorting import NonDominatedSorting


class Algorithm:
    """

    This class represents the abstract class for any algorithm to be implemented. Most importantly it
    provides the solve method that is used to optimize a given problem.

    The solve method provides a wrapper function which does validate the input.

    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.disp = None
        self.func_display_attrs = None
        self.callback = None
        self.history = []
        self.save_history = None
        self.pf = None

    def solve(self,
              problem,
              termination,
              seed=1,
              disp=False,
              callback=None,
              save_history=False,
              pf=None,
              **kwargs
              ):
        """

        Solve a given problem by a given evaluator. The evaluator determines the termination condition and
        can either have a maximum budget, hypervolume or whatever. The problem can be any problem the algorithm
        is able to solve.

        Parameters
        ----------

        problem: class
            Problem to be solved by the algorithm

        termination: class
            object that evaluates and saves the number of evaluations and determines the stopping condition

        seed: int
            Random seed for this run. Before the algorithm starts this seed is set.

        disp : bool
            If it is true than information during the algorithm execution are displayed

        callback : func
            A callback function can be passed that is executed every generation. The parameters for the function
            are the algorithm itself, the number of evaluations so far and the current population.

                def callback(algorithm):
                    pass

        save_history : bool
            If true, a current snapshot of each generation is saved.

        pf : np.array
            The Pareto-front for the given problem. If provided performance metrics are printed during execution.

        Returns
        -------
        res : dict
            A dictionary that saves all the results of the algorithm. Also, the history if save_history is true.

        """

        # set the random seed for generator
        random.seed(seed)

        self.disp = disp
        self.callback = callback
        self.save_history = save_history
        self.pf = pf

        if isinstance(termination, int):
            termination = MaximumFunctionCallTermination(termination)

        # call the algorithm to solve the problem
        pop = self._solve(problem, termination)

        # get the optimal result by filtering feasible and non-dominated
        opt = pop.copy()
        opt = opt[opt.collect(lambda ind: ind.feasible)[:, 0]]

        # if at least one feasible solution was found
        if len(opt) > 0:

            if problem.n_obj > 1:
                I = NonDominatedSorting().do(opt.get("F"), only_non_dominated_front=True)
                opt = opt[I]
                X, F, CV, G = opt.get("X", "F", "CV", "G")

            else:
                opt = pop[np.argmin(pop.get("F"))]
                X, F, CV, G = opt.X, opt.F, opt.CV, opt.G
        else:
            opt = None

        res = Result(opt, opt is None, "")
        res.algorithm, res.problem, res.pf = self, problem, pf
        res.pop = pop

        if opt is not None:
            res.X, res.F, res.CV, res.G = X, F, CV, G

        res.history = self.history

        return res

    # method that is called each iteration to call so#me methods regularly
    def _each_iteration(self, D, first=False, **kwargs):

        # display the output if defined by the algorithm
        if self.disp and self.func_display_attrs is not None:
            disp = self.func_display_attrs(D['problem'], D['evaluator'], D, self.pf)
            if disp is not None:
                self._display(disp, header=first)

        # if a callback function is provided it is called after each iteration
        if self.callback is not None:
            self.callback(self)

        if self.save_history:
            hist = self.history
            self.history = None

            obj = copy.deepcopy(self)
            self.history = hist

            self.history.append(obj)

    # attributes are a list of tuples of length 3: (name, val, width)
    def _display(self, disp, header=False):
        regex = " | ".join(["{}"] * len(disp))
        if header:
            print("=" * 50)
            print(regex.format(*[name.ljust(width) for name, _, width in disp]))
            print("=" * 50)
        print(regex.format(*[str(val).ljust(width) for _, val, width in disp]))

    @abstractmethod
    def _solve(self, problem, termination):
        pass
