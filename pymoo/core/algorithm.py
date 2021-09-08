import copy
import random
import time

import numpy as np

from pymoo.core.callback import Callback
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.result import Result
from pymoo.util.function_loader import FunctionLoader
from pymoo.util.misc import termination_from_tuple
from pymoo.util.optimum import filter_optimum


class Algorithm:
    """

    This class represents the abstract class for any algorithm to be implemented. Most importantly it
    provides the solve method that is used to optimize a given problem.

    The solve method provides a wrapper function which does validate the input.


    Parameters
    ----------

    problem : :class:`~pymoo.core.problem.Problem`
        Problem to be solved by the algorithm

    termination: :class:`~pymoo.core.termination.Termination`
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

    evaluator : :class:`~pymoo.core.evaluator.Evaluator`
        The evaluator which can be used to make modifications before calling the evaluate function of a problem.


    """

    def __init__(self, **kwargs):

        # !
        # Here all algorithm parameters needed no matter what is problem is passed are defined
        # problem dependent initialization happens in setup(problem, **kwargs)
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
        # a placeholder object for implementation to store solutions in each iteration
        self.off = None
        # the optimum found by the algorithm
        self.opt = None
        # can be used to store additional data in submodules
        self.data = {}
        # if the initialized method has been called before or not
        self.is_initialized = False
        # the time when the algorithm has been setup for the first time
        self.start_time = None

    # =========================================================================================================
    # PUBLIC
    # =========================================================================================================

    def setup(self,
              problem,

              # START Overwrite by minimize
              termination=None,
              callback=None,
              display=None,
              # END Overwrite by minimize

              # START Default minimize
              seed=None,
              verbose=False,
              save_history=False,
              return_least_infeasible=False,
              # END Default minimize

              pf=True,
              evaluator=None,
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
            self.termination = termination_from_tuple(termination)
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

        # no call the algorithm specific setup given the problem
        self._setup(problem, **kwargs)

        return self

    def run(self):

        # now the termination criterion should be set
        if self.termination is None:
            raise Exception("No termination criterion defined and algorithm has no default termination implemented!")

        # while termination criterion not fulfilled
        while self.has_next():
            self.next()

        # create the result object to be returned
        res = self.result()

        return res

    def has_next(self):
        return not self.has_terminated

    def finalize(self):
        return self._finalize()

    def next(self):

        # get the infill solutions
        infills = self.infill()

        # call the advance with them after evaluation
        if infills is not None:
            self.evaluator.eval(self.problem, infills, algorithm=self)
            self.advance(infills=infills)

        # if the algorithm does not follow the infill-advance scheme just call advance
        else:
            self.advance()

    def _initialize(self):

        # the time starts whenever this method is called
        self.start_time = time.time()

        # set the attribute for the optimization method to start
        self.n_gen = 1
        self.has_terminated = False
        self.pop, self.opt = Population(), None

        # if the history is supposed to be saved
        if self.save_history:
            self.history = []

    def infill(self):
        if self.problem is None:
            raise Exception("Please call `setup(problem)` before calling next().")

        # the first time next is called simply initial the algorithm - makes the interface cleaner
        if not self.is_initialized:

            # hook mostly used by the class to happen before even to initialize
            self._initialize()

            # execute the initialization infill of the algorithm
            infills = self._initialize_infill()

        else:
            # request the infill solutions if the algorithm has implemented it
            infills = self._infill()

        # set the current generation to the offsprings
        if infills is not None:
            infills.set("n_gen", self.n_gen)

        return infills

    def advance(self, infills=None, **kwargs):

        # if infills have been provided set them as offsprings and feed them into advance
        self.off = infills

        # if the algorithm has not been already initialized
        if not self.is_initialized:

            # set the generation counter to 1
            self.n_gen = 1

            # assign the population to the algorithm
            self.pop = infills

            # do whats necessary after the initialization
            self._initialize_advance(infills=infills, **kwargs)

            # set this algorithm to be initialized
            self.is_initialized = True

        else:

            # increase the generation counter by one
            self.n_gen += 1

            # call the implementation of the advance method - if the infill is not None
            self._advance(infills=infills, **kwargs)

        # execute everything which needs to be done after having the algorithm advanced to the next generation
        self._post_advance()

        # set whether the algorithm has terminated or not
        self.has_terminated = not self.termination.do_continue(self)

        # if the algorithm has terminated call the finalize method
        if self.has_terminated:
            return self.finalize()

    def result(self):
        res = Result()

        # store the time when the algorithm as finished
        res.start_time = self.start_time
        res.end_time = time.time()
        res.exec_time = res.end_time - res.start_time

        res.pop = self.pop

        # get the optimal solution found
        opt = self.opt
        if opt is None or len(opt) == 0:
            opt = None

        # if no feasible solution has been found
        elif not np.any(opt.get("feasible")):
            if self.return_least_infeasible:
                opt = filter_optimum(opt, least_infeasible=True)
            else:
                opt = None
        res.opt = opt

        # if optimum is set to none to not report anything
        if res.opt is None:
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

    def ask(self, *args, **kwargs):
        return self.infill(*args, **kwargs)

    def tell(self, *args, **kwargs):
        return self.advance(*args, **kwargs)

    # =========================================================================================================
    # PROTECTED
    # =========================================================================================================

    def _set_optimum(self):
        self.opt = filter_optimum(self.pop, least_infeasible=True)

    def _post_advance(self):

        # update the current optimum of the algorithm
        self._set_optimum()

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
            _hist, _callback = self.history, self.callback

            self.history, self.callback = None, None
            obj = copy.deepcopy(self)

            self.history, self.callback = _hist, _callback
            self.history.append(obj)

    # =========================================================================================================
    # TO BE OVERWRITTEN
    # =========================================================================================================

    def _setup(self, problem, **kwargs):
        pass

    def _initialize_infill(self):
        pass

    def _initialize_advance(self, infills=None, **kwargs):
        pass

    def _infill(self):
        pass

    def _advance(self, infills=None, **kwargs):
        pass

    def _finalize(self):
        pass
