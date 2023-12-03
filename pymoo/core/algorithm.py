from typing import Generator

from numpy.random import RandomState

from pymoo.core.archive import Optimum
from pymoo.core.display import Display
from pymoo.core.evaluator import Evaluator, Evaluation
from pymoo.core.observer import Observable
from pymoo.core.output import Output
from pymoo.core.parallelize import Parallelize, Serial
from pymoo.core.problem import Problem
from pymoo.core.solution import SolutionSet, Solution
from pymoo.core.termination import Termination


class Algorithm(Observable):

    def __init__(self,
                 termination: Termination = Termination(),
                 display: Display = Display(),
                 output: Output = None,
                 ):
        super().__init__()

        self.problem = None
        self.evaluator = None
        self.verbose = None
        self.seed = None
        self.random_state = None

        self.is_initialized = False
        self.iter = 1

        self.termination = termination
        self.output = output
        self.display = display
        self.observers = [self.termination, self.display]

        self.opt = Optimum()
        self.sols = None

        self.generator = None
        self.cache = None

    def setup(self,
              problem: Problem,
              parallelize: Parallelize = Serial(),
              seed: int = None,
              random_state: RandomState = None,
              verbose: bool = False,
              **kwargs):

        # set the problem and define the evaluator object
        self.problem = problem
        self.evaluator = Evaluator(problem, parallelize)

        # define the random seed of a run. This helps to reproduce a run again.
        if random_state is None:
            if seed is None:
                seed = RandomState().randint(0, 1_000_000)
            self.seed = seed

            random_state = RandomState(seed)

        # create a random state for all actions that require randomness
        self.random_state = random_state

        # whether output should be displayed
        self.verbose = verbose

        # trigger the setup event in the very end
        self.event('setup')

        return self

    def run(self):
        while self.has_next():
            self.next()
        self.finalize()
        return self.opt.solution()

    def has_next(self):
        return self.termination.has_terminated()

    def next(self):
        self.start_step()
        self.step()
        self.end_step()
        return self.sols

    def start_step(self):
        self.generator = self.advance() if self.is_initialized else self.initialize()

    def step(self):
        while self.has_evaluation():
            evaluation = self.ask()
            evaluation.run()
            self.tell(evaluation)

    def end_step(self):

        sols = self.cache
        self.cache = None

        self.sols = sols
        self.opt.add(sols)

        # send the event for the current iteration
        if not self.is_initialized:
            self.event('initialize')
            self.is_initialized = True
        else:
            self.event('update')

        self.iter += 1

    def has_evaluation(self):
        return self.generator is not None

    def ask(self) -> Evaluation:

        if self.cache is None:
            self.cache = next(self.generator)

        return self.cache

    def tell(self, evaluation: Evaluation):
        try:
            self.cache = self.generator.send(evaluation)
        except StopIteration as e:
            self.generator = None
            self.cache = e.value

    def initialize(self) -> Generator[Evaluation, Solution | SolutionSet, SolutionSet]:
        sols = yield from self.advance()
        return sols

    def advance(self) -> Generator[Evaluation, Solution | SolutionSet, SolutionSet]:
        pass

    def finalize(self):
        self.event('finalize')
