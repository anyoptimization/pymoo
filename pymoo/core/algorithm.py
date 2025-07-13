import copy
import time

import numpy as np

from pymoo.core.callback import Callback
from pymoo.core.evaluator import Evaluator
from pymoo.core.meta import Meta
from pymoo.core.population import Population
from pymoo.core.result import Result
from pymoo.functions import FunctionLoader
from pymoo.termination.default import DefaultMultiObjectiveTermination, DefaultSingleObjectiveTermination
from pymoo.util.display.display import Display
from pymoo.util.misc import termination_from_tuple
from pymoo.util.optimum import filter_optimum


class Algorithm:

    def __init__(self,
                 termination=None,
                 output=None,
                 display=None,
                 callback=None,
                 archive=None,
                 return_least_infeasible=False,
                 save_history=False,
                 verbose=False,
                 seed=None,
                 evaluator=None,
                 **kwargs):

        super().__init__()

        # prints the compile warning if enabled
        FunctionLoader.get_instance()

        # the problem to be solved (will be set later on)
        self.problem = None

        # the termination criterion to be used by the algorithm - might be specific for an algorithm
        self.termination = termination

        # the text that should be printed during the algorithm run
        self.output = output

        # an archive kept during algorithm execution (not always the same as optimum)
        self.archive = archive

        # the form of display shown during algorithm execution
        self.display = display

        # callback to be executed each generation
        if callback is None:
            callback = Callback()
        self.callback = callback

        # whether the algorithm should finally return the least infeasible solution if no feasible found
        self.return_least_infeasible = return_least_infeasible

        # whether the history should be saved or not
        self.save_history = save_history

        # whether the algorithm should print output in this run or not
        self.verbose = verbose

        # the random seed that was used
        self.seed = seed
        self.random_state = None

        # the function evaluator object (can be used to inject code)
        if evaluator is None:
            evaluator = Evaluator()
        self.evaluator = evaluator

        # the history object which contains the list
        self.history = list()

        # the current solutions stored - here considered as population
        self.pop = None

        # a placeholder object for implementation to store solutions in each iteration
        self.off = None

        # the optimum found by the algorithm
        self.opt = None

        # the current number of generation or iteration
        self.n_iter = None

        # can be used to store additional data in submodules
        self.data = {}

        # if the initialized method has been called before or not
        self.is_initialized = False

        # the time when the algorithm has been setup for the first time
        self.start_time = None

    def setup(self, problem, verbose=False, progress=False, **kwargs):

        # the problem to be solved by the algorithm
        self.problem = problem

        # clone the output object if it exists to avoid state pollution between runs
        if self.output is not None:
            self.output = copy.deepcopy(self.output)

        # set all the provided options to this method
        for key, value in kwargs.items():
            self.__dict__[key] = value

        # set random state
        self.random_state = np.random.default_rng(self.seed)

        # make sure that some type of termination criterion is set
        if self.termination is None:
            self.termination = default_termination(problem)
        else:
            self.termination = termination_from_tuple(self.termination)

        # set up the display during the algorithm execution
        if self.display is None:
            self.display = Display(self.output, verbose=verbose, progress=progress)

        # finally call the function that can be overwritten by the actual algorithm
        self._setup(problem, **kwargs)

        return self

    def run(self):
        while self.has_next():
            self.next()
        return self.result()

    def has_next(self):
        return not self.termination.has_terminated()

    def finalize(self):

        # finalize the display output in the end of the run
        self.display.finalize()

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
        self.n_iter = 1
        self.pop = Population.empty()
        self.opt = None

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
            infills.set("n_gen", self.n_iter)
            infills.set("n_iter", self.n_iter)

        return infills

    def advance(self, infills=None, **kwargs):

        # if infills have been provided set them as offsprings and feed them into advance
        self.off = infills

        # if the algorithm has not been already initialized
        if not self.is_initialized:

            # set the generation counter to 1
            self.n_iter = 1

            # assign the population to the algorithm
            self.pop = infills

            # do what is necessary after the initialization
            self._initialize_advance(infills=infills, **kwargs)

            # set this algorithm to be initialized
            self.is_initialized = True

            # always advance to the next iteration after initialization
            self._post_advance()

        else:

            # call the implementation of the advance method - if the infill is not None
            val = self._advance(infills=infills, **kwargs)

            # always advance to the next iteration - except if the algorithm returns False
            if val is None or val:
                self._post_advance()

        # if the algorithm has terminated, then do the finalization steps and return the result
        if self.termination.has_terminated():
            self.finalize()
            ret = self.result()

        # otherwise just increase the iteration counter for the next step and return the current optimum
        else:
            ret = self.opt

        # add the infill solutions to an archive
        if self.archive is not None and infills is not None:
            self.archive = self.archive.add(infills)

        return ret

    def result(self):
        res = Result()

        # store the time when the algorithm as finished
        res.start_time = self.start_time
        res.end_time = time.time()
        res.exec_time = res.end_time - res.start_time

        res.pop = self.pop
        res.archive = self.archive
        res.data = self.data

        # get the optimal solution found
        opt = self.opt
        if opt is None or len(opt) == 0:
            opt = None

        # if no feasible solution has been found
        elif not np.any(opt.get("FEAS")):
            if self.return_least_infeasible:
                opt = filter_optimum(opt, least_infeasible=True)
            else:
                opt = None
        res.opt = opt

        # if optimum is set to none to not report anything
        if res.opt is None:
            X, F, CV, G, H = None, None, None, None, None

        # otherwise get the values from the population
        else:
            X, F, CV, G, H = self.opt.get("X", "F", "CV", "G", "H")

            # if single-objective problem and only one solution was found - create a 1d array
            if self.problem.n_obj == 1 and len(X) == 1:
                X, F, CV, G, H = X[0], F[0], CV[0], G[0], H[0]

        # set all the individual values
        res.X, res.F, res.CV, res.G, res.H = X, F, CV, G, H

        # create the result object
        res.problem = self.problem
        res.history = self.history

        return res

    def ask(self):
        return self.infill()

    def tell(self, *args, **kwargs):
        return self.advance(*args, **kwargs)

    def _set_optimum(self):
        self.opt = filter_optimum(self.pop, least_infeasible=True)

    def _post_advance(self):

        # update the current optimum of the algorithm
        self._set_optimum()

        # update the current termination condition of the algorithm
        self.termination.update(self)

        # display the output if defined by the algorithm
        self.display(self)

        if self.save_history:
            _hist, _callback, _display = self.history, self.callback, self.display

            self.history, self.callback, self.display = None, None, None
            obj = copy.deepcopy(self)

            self.history, self.callback, self.display = _hist, _callback, _display
            self.history.append(obj)

        # if a callback function is provided it is called after each iteration
        self.callback(self)

        self.n_iter += 1

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

    # =========================================================================================================
    # CONVENIENCE
    # =========================================================================================================

    @property
    def n_gen(self):
        return self.n_iter

    @n_gen.setter
    def n_gen(self, value):
        self.n_iter = value


class LoopwiseAlgorithm(Algorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generator = None
        self.state = None

    def _next(self):
        pass

    def _infill(self):
        if self.state is None:
            self._advance()
        return self.state

    def _advance(self, infills=None, **kwargs):
        if self.generator is None:
            self.generator = self._next()
        try:
            self.state = self.generator.send(infills)
        except StopIteration:
            self.generator = None
            self.state = None
            return True

        return False


def default_termination(problem):
    if problem.n_obj > 1:
        termination = DefaultMultiObjectiveTermination()
    else:
        termination = DefaultSingleObjectiveTermination()
    return termination


class MetaAlgorithm(Meta):
    """
    An algorithm wrapper that combines Algorithm's functionality with Meta's delegation behavior.
    Uses Meta to provide transparent proxying with the ability to override specific methods.
    """

    def __init__(self, algorithm, copy=True, **kwargs):
        # If the algorithm is already a Meta object, don't copy to avoid deepcopy issues with nested proxies
        if isinstance(algorithm, Meta):
            copy = False
            
        # Initialize Meta (which initializes wrapt.ObjectProxy)
        super().__init__(algorithm, copy=copy)
        
        # Pass any additional kwargs to the wrapped algorithm if needed
        for key, value in kwargs.items():
            setattr(self, key, value)
