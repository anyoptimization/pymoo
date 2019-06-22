import copy
from abc import abstractmethod

import numpy as np

from pymoo.model.evaluator import Evaluator
from pymoo.model.result import Result
from pymoo.util.function_loader import FunctionLoader

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class Algorithm:
    """

    This class represents the abstract class for any algorithm to be implemented. Most importantly it
    provides the solve method that is used to optimize a given problem.

    The solve method provides a wrapper function which does validate the input.

    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.evaluator = None
        self.problem = None
        self.termination = None
        self.pf = None
        self.opt = None
        self.seed = None

        self.history = None
        self.verbose = None
        self.func_display_attrs = None
        self.callback = None
        self.save_history = None

    def solve(self,
              problem,
              termination,
              seed=None,
              verbose=False,
              callback=None,
              save_history=False,
              pf=None,
              evaluator=None,
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

        verbose : bool
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

        evaluator : class
            The evaluator which can be used to make modifications before calling the evaluate function of a problem.

        Returns
        -------
        res : dict
            A dictionary that saves all the results of the algorithm. Also, the history if save_history is true.

        """

        # prints the compile warning if enabled
        FunctionLoader.get_instance()

        # set the random seed for generator and save if
        if seed is not None:
            np.random.seed(seed)
        else:
            seed = np.random.randint(0, 10000000)
            np.random.seed(seed)
        self.seed = seed

        # the evaluator object which is counting the evaluations
        self.history = []

        self.evaluator = evaluator
        if self.evaluator is None:
            self.evaluator = Evaluator()

        self.problem = problem

        self.termination = termination
        if self.termination is not None:
            self.termination.reset()

        self.pf = pf

        self.verbose = verbose
        self.callback = callback
        self.save_history = save_history

        # call the algorithm to solve the problem
        pop = self._solve(problem, termination)

        # get the optimal result by filtering feasible and non-dominated
        if self.opt is None:
            opt = pop.copy()
        else:
            opt = self.opt

        I = opt.collect(lambda ind: ind.feasible)
        
        # if at least one feasible solution was found
        if I.size != 0:
            opt = opt[I[:, 0]]

            if problem.n_obj > 1:
                I = NonDominatedSorting().do(opt.get("F"), only_non_dominated_front=True)
                opt = opt[I]
                X, F, CV, G = opt.get("X", "F", "CV", "G")

            else:
                opt = opt[np.argmin(opt.get("F"))]
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

    def next(self, pop):
        return self._next(pop)

    def initialize(self, pop):
        return self._initialize()

    # method that is called each iteration to call so#me methods regularly
    def _each_iteration(self, D, first=False, **kwargs):

        # display the output if defined by the algorithm
        if self.verbose and self.func_display_attrs is not None:
            disp = self.func_display_attrs(self.problem, self.evaluator, self, self.pf)
            if disp is not None:
                self._display(disp, header=first)

        # if a callback function is provided it is called after each iteration
        if self.callback is not None:
            # use the callback here without having the function itself
            self.callback(self)

        if self.save_history:
            hist, _callback = self.history, self.callback
            self.history, self.callback = None, None

            obj = copy.deepcopy(self)
            self.history = hist
            self.callback = _callback

            self.history.append(obj)

    # attributes are a list of tuples of length 3: (name, val, width)
    def _display(self, disp, header=False):
        regex = " | ".join(["{}"] * len(disp))
        if header:
            print("=" * 70)
            print(regex.format(*[name.ljust(width) for name, _, width in disp]))
            print("=" * 70)
        print(regex.format(*[str(val).ljust(width) for _, val, width in disp]))

    @abstractmethod
    def _solve(self, problem, termination):
        pass
