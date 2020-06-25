import copy
import random
import time
from abc import abstractmethod

import numpy as np

from pymoo.model.callback import Callback
from pymoo.model.evaluator import Evaluator
from pymoo.model.individual import Individual
from pymoo.model.population import Population
from pymoo.model.result import Result
from pymoo.util.function_loader import FunctionLoader
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class Algorithm:
    """

    This class represents the abstract class for any algorithm to be implemented. Most importantly it
    provides the solve method that is used to optimize a given problem.

    The solve method provides a wrapper function which does validate the input.


    Parameters
    ----------

    problem : :class:`~pymoo.model.problem.Problem`
        Problem to be solved by the algorithm

    termination: :class:`~pymoo.model.termination.Termination`
        Object that tells the algorithm when to terminate.

    seed : int
        Random seed to be used. Same seed is supposed to return the same result. If set to None, a random seed
        is chosen randomly and stored in the result object to ensure reproducibility.

    verbose : bool
        If true information during the algorithm execution are displayed

    callback : func
        A callback function can be passed that is executed every generation. The parameters for the function
        are the algorithm itself, the number of evaluations so far and the current population.

            def callback(algorithm):
                pass

    save_history : bool
        If true, a current snapshot of each generation is saved.

    pf : numpy.array
        The Pareto-front for the given problem. If provided performance metrics are printed during execution.

    return_least_infeasible : bool
        Whether the algorithm should return the least infeasible solution, if no solution was found.

    evaluator : :class:`~pymoo.model.evaluator.Evaluator`
        The evaluator which can be used to make modifications before calling the evaluate function of a problem.


    """

    def __init__(self, **kwargs):

        # !
        # Here all algorithm parameters needed no matter what is problem is passed are defined
        # problem dependent initialization happens in initialize(problem, **kwargs)
        # !

        super().__init__()

        # prints the compile warning if enabled
        FunctionLoader.get_instance()

        # !
        # DEFAULT SETTINGS OF ALGORITHM
        # !
        # the termination criterion to be used by the algorithm - might be specific for an algorithm
        self.termination = kwargs.get("termination")
        # set the display variable supplied to the algorithm - might be specific for an algorithm
        self.display = kwargs.get("display")
        # callback to be executed each generation
        self.callback = kwargs.get("callback")

        # !
        # Attributes to be set later on for each problem run
        # !
        # the optimization problem as an instance
        self.problem = None
        # whether the algorithm should finally return the least infeasible solution if no feasible found
        self.return_least_infeasible = None
        # whether the history should be saved or not
        self.save_history = None
        # whether the algorithm should print output in this run or not
        self.verbose = None
        # the random seed that was used
        self.seed = None
        # an algorithm can defined the default termination which can be overwritten
        self.default_termination = None
        # whether the algorithm as terminated or not
        self.has_terminated = None
        # the pareto-front of the problem - if it exist or passed
        self.pf = None
        # the function evaluator object (can be used to inject code)
        self.evaluator = None
        # the current number of generation or iteration
        self.n_gen = None
        # the history object which contains the list
        self.history = None
        # the current solutions stored - here considered as population
        self.pop = None
        # the optimum found by the algorithm
        self.opt = None
        # can be used to store additional data in submodules
        self.data = {}

    # =========================================================================================================
    # PUBLIC
    # =========================================================================================================

    def initialize(self,
                   problem,
                   pf=True,
                   evaluator=None,

                   # START Default minimize
                   seed=None,
                   verbose=False,
                   save_history=False,
                   return_least_infeasible=False,
                   # END Default minimize

                   # START Overwrite by minimize
                   termination=None,
                   callback=None,
                   display=None,
                   # END Overwrite by minimize

                   **kwargs):

        # set the problem that is optimized for the current run
        self.problem = problem

        # set the provided pareto front
        self.pf = pf

        # by default make sure an evaluator exists if nothing is passed
        if evaluator is None:
            evaluator = Evaluator()
        self.evaluator = evaluator

        # !
        # START Default minimize
        # !
        # if this run should be verbose or not
        self.verbose = verbose
        # whether the least infeasible should be returned or not
        self.return_least_infeasible = return_least_infeasible
        # whether the history should be stored or not
        self.save_history = save_history

        # set the random seed in the algorithm object
        self.seed = seed
        if self.seed is None:
            self.seed = np.random.randint(0, 10000000)
        # set the random seed for Python and Numpy methods
        random.seed(self.seed)
        np.random.seed(self.seed)
        # !
        # END Default minimize
        # !

        # !
        # START Overwrite by minimize
        # !
        # the termination criterion to be used to stop the algorithm
        if self.termination is None:
            self.termination = termination
        # if nothing given fall back to default
        if self.termination is None:
            self.termination = self.default_termination

        if callback is not None:
            self.callback = callback
        if display is not None:
            self.display = display

        # !
        # END Overwrite by minimize
        # !

        # by default the algorithm as not terminated
        self.has_terminated = False

        # other run dependent variables that are reset
        self.n_gen = 1
        self.history = []
        self.pop = Population()
        self.opt = None

    def solve(self):

        # the result object to be finally returned
        res = Result()

        # set the timer in the beginning of the call
        res.start_time = time.time()

        # call the algorithm to solve the problem
        self._solve(self.problem)

        # store the time when the algorithm as finished
        res.end_time = time.time()
        res.exec_time = res.end_time - res.start_time

        # store the resulting population
        res.pop = self.pop

        # get the optimal solution found
        opt = self.opt

        # if optimum is not set
        if len(opt) == 0:
            opt = None

        # if no feasible solution has been found
        elif not np.any(opt.get("feasible")):
            if self.return_least_infeasible:
                opt = filter_optimum(opt, least_infeasible=True)
            else:
                opt = None

        # set the optimum to the result object
        res.opt = opt

        # if optimum is set to none to not report anything
        if opt is None:
            X, F, CV, G = None, None, None, None

        # otherwise get the values from the population
        else:
            X, F, CV, G = self.opt.get("X", "F", "CV", "G")

            # if single-objective problem and only one solution was found - create a 1d array
            if self.problem.n_obj == 1 and len(X) == 1:
                X, F, CV, G = X[0], F[0], CV[0], G[0]

        # set all the individual values
        res.X, res.F, res.CV, res.G = X, F, CV, G

        # create the result object
        res.problem, res.pf = self.problem, self.pf
        res.history = self.history

        return res

    def next(self):
        # increase the generation counter
        self.n_gen += 1

        # call next of the implementation of the algorithm
        self._next()

        # set the optimum - only done if the algorithm did not do it yet
        self._set_optimum()

        # do what needs to be done each generation
        self._each_iteration()

    def finalize(self):
        return self._finalize()

    # =========================================================================================================
    # PROTECTED
    # =========================================================================================================

    def _solve(self, problem):

        # now the termination criterion should be set
        if self.termination is None:
            raise Exception("No termination criterion defined and algorithm has no default termination implemented!")

        # initialize the first population and evaluate it
        self._initialize()
        self._set_optimum()
        self._each_iteration()

        # while termination criterion not fulfilled
        while self.termination.do_continue(self):
            self.next()

        # set the algorithm to be terminated from now on
        self.has_terminated = True

        # finalize the algorithm and do postprocessing of desired
        self.finalize()

    # method that is called each iteration to call some algorithms regularly
    def _each_iteration(self, *args, **kwargs):

        # display the output if defined by the algorithm
        if self.verbose and self.display is not None:
            self.display.do(self.problem, self.evaluator, self, pf=self.pf)

        # if a callback function is provided it is called after each iteration
        if self.callback is not None:
            if isinstance(self.callback, Callback):
                self.callback.notify(self)
            else:
                self.callback(self)

        if self.save_history:
            hist, _callback = self.history, self.callback
            self.history, self.callback = None, None

            obj = copy.deepcopy(self)
            self.history = hist
            self.callback = _callback

            self.history.append(obj)

    def _set_optimum(self, force=False):
        pop = self.pop
        # if self.opt is not None:
        #     pop = Population.merge(pop, self.opt)
        self.opt = filter_optimum(pop, least_infeasible=True)

    def _finalize(self):
        pass

    @abstractmethod
    def _initialize(self):
        pass

    @abstractmethod
    def _next(self):
        pass


def filter_optimum(pop, least_infeasible=False):
    # first only choose feasible solutions
    ret = pop[pop.get("feasible")[:, 0]]

    # if at least one feasible solution was found
    if len(ret) > 0:

        # then check the objective values
        F = ret.get("F")

        if F.shape[1] > 1:
            I = NonDominatedSorting().do(F, only_non_dominated_front=True)
            ret = ret[I]

        else:
            ret = ret[np.argmin(F)]

    # no feasible solution was found
    else:
        # if flag enable report the least infeasible
        if least_infeasible:
            ret = pop[np.argmin(pop.get("CV"))]
        # otherwise just return none
        else:
            ret = None

    if isinstance(ret, Individual):
        ret = Population().create(ret)

    return ret
