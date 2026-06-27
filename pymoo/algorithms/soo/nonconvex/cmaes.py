"""Covariance Matrix Adaptation Evolution Strategy (CMA-ES)."""

import cma
import numpy as np

from pymoo.algorithms.base.local import LocalSearch
from pymoo.core.population import Population
from pymoo.core.termination import NoTermination
from pymoo.docs import parse_doc_string
from pymoo.termination.max_eval import MaximumFunctionCallTermination
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.util.display.column import Column
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.util.normalization import ZeroToOneNormalization, NoNormalization
from pymoo.util.optimum import filter_optimum
from pymoo.vendor.vendor_cmaes import my_fmin


# =========================================================================================================
# Implementation
# =========================================================================================================


class CMAESOutput(SingleObjectiveOutput):
    def __init__(self):
        super().__init__()

        self.sigma = Column("sigma")
        self.min_std = Column("min_std", width=8)
        self.max_std = Column("max_std", width=8)
        self.axis = Column("axis", width=8)

        self.run = Column("run", width=4)
        self.fpop = Column("fpop", width=8)
        self.n_pop = Column("n_pop", width=5)

    def initialize(self, algorithm):
        super().initialize(algorithm)

        if algorithm.restarts > 0:
            self.columns += [self.run, self.fpop, self.n_pop]

        self.columns += [self.sigma, self.min_std, self.max_std, self.axis]

    def update(self, algorithm):
        super().update(algorithm)

        if not algorithm.es.gi_frame:
            return

        fmin = algorithm.es.gi_frame.f_locals
        cma = fmin["es"]

        self.sigma.set(cma.sigma)

        val = cma.sigma_vec * cma.dC**0.5
        self.min_std.set((cma.sigma * min(val)))
        self.max_std.set((cma.sigma * max(val)))

        if algorithm.restarts > 0:
            self.run.set(int(fmin["irun"] - fmin["runs_with_small"]) + 1)
            self.fpop.set(algorithm.pop.get("F").min())
            self.n_pop.set(int(cma.opts["popsize"]))

        axis = (
            cma.D.max() / cma.D.min()
            if not cma.opts["CMA_diagonal"] or cma.countiter > cma.opts["CMA_diagonal"]
            else max(cma.sigma_vec * 1) / min(cma.sigma_vec * 1)
        )
        self.axis.set(axis)


class CMAES(LocalSearch):
    def __init__(
        self,
        x0=None,
        sigma=0.1,
        normalize=True,
        parallelize=True,
        maxfevals=np.inf,
        tolfun=1e-11,
        tolx=1e-11,
        restarts=0,
        restart_from_best="False",
        incpopsize=2,
        eval_initial_x=False,
        noise_handler=None,
        noise_change_sigma_exponent=1,
        noise_kappa_exponent=0,
        bipop=False,
        cmaes_verbose=-9,
        verb_log=0,
        output=CMAESOutput(),
        pop_size=None,
        **kwargs,
    ):
        """Covariance Matrix Adaptation Evolution Strategy.

        Args:
            x0: Initial guess of minimum solution (array or string expression).
            sigma: Initial standard deviation in each coordinate.
            normalize: Whether to normalize problem bounds.
            parallelize: Whether to call objective function batch-wise.
            maxfevals: Maximum number of function evaluations.
            tolfun: Termination tolerance in function value.
            tolx: Termination tolerance in x-changes.
            restarts: Number of restarts with increasing population size (IPOP-CMA-ES).
            restart_from_best: Whether to restart from best solution.
            incpopsize: Multiplier for population size increase.
            eval_initial_x: Whether to evaluate initial solution.
            noise_handler: Noise handling instance or class.
            noise_change_sigma_exponent: Exponent for sigma increment.
            noise_kappa_exponent: Kappa exponent for noise treatment.
            bipop: Whether to use BIPOP-CMA-ES restart strategy.
            cmaes_verbose: Verbosity level for CMA-ES output.
            verb_log: Verbosity for logging to files.
            output: Output display configuration.
            pop_size: Population size (overrides CMA-ES default).
            **kwargs: Additional CMA-ES options passed to CMAEvolutionStrategy.
        """
        if pop_size is not None:
            parallelize = True
            kwargs["popsize"] = pop_size

        super().__init__(x0=x0, output=output, **kwargs)

        self.termination = NoTermination()

        self.es = None
        self.cma = None

        self.normalize = normalize
        self.norm = None

        self.sigma = sigma
        self.restarts = restarts
        self.restart_from_best = restart_from_best
        self.incpopsize = incpopsize
        self.eval_initial_x = eval_initial_x
        self.noise_handler = noise_handler
        self.noise_change_sigma_exponent = noise_change_sigma_exponent
        self.noise_kappa_exponent = noise_kappa_exponent
        self.bipop = bipop

        self.options = dict(
            verbose=cmaes_verbose,
            verb_log=verb_log,
            maxfevals=maxfevals,
            tolfun=tolfun,
            tolx=tolx,
            **kwargs,
        )

        self.send_array_to_yield = True
        self.parallelize = parallelize
        self.al = None

    def _setup(self, problem, **kwargs):

        xl, xu = problem.bounds()
        if self.normalize:
            self.norm, self.options["bounds"] = bounds_if_normalize(xl, xu)
        else:
            self.norm = NoNormalization()
            self.options["bounds"] = [xl, xu]

        seed = kwargs.get("seed", self.seed)
        self.options["seed"] = seed

        if isinstance(self.termination, MaximumGenerationTermination):
            self.options["maxiter"] = self.termination.n_max_gen
        elif isinstance(self.termination, MaximumFunctionCallTermination):
            self.options["maxfevals"] = self.termination.n_max_evals

    def _initialize_advance(self, **kwargs):
        super()._initialize_advance(**kwargs)

        kwargs = dict(
            options=self.options,
            parallel_objective=self.parallelize,
            restarts=self.restarts,
            restart_from_best=self.restart_from_best,
            incpopsize=self.incpopsize,
            eval_initial_x=self.eval_initial_x,
            noise_handler=self.noise_handler,
            noise_change_sigma_exponent=self.noise_change_sigma_exponent,
            noise_kappa_exponent=self.noise_kappa_exponent,
            bipop=self.bipop,
            random_state=self.random_state,
        )

        x0 = self.norm.forward(self.x0.X)
        self.es = my_fmin(x0, self.sigma, **kwargs)

        # do this to allow the printout in the first generation
        self.next_X = next(self.es)

    def _infill(self):

        X = np.array(self.next_X)
        self.send_array_to_yield = X.ndim > 1
        X = np.atleast_2d(X)

        # evaluate the population
        self.pop = Population.new("X", self.norm.backward(X))

        return self.pop

    def _advance(self, infills=None, **kwargs):

        if infills is None:
            self.termination.force_termination = True

        else:
            # set infeasible individual's objective values to np.nan - then CMAES can handle it
            for ind in infills:
                if not ind.feas:
                    ind.F[:] = np.inf

            F = infills.get("f").tolist()
            if not self.send_array_to_yield:
                F = F[0]

            try:
                self.next_X = self.es.send(F)
            except:  # noqa: E722
                self.next_X = None

            if self.next_X is None:
                self.termination.force_termination = True

    def _set_optimum(self):
        pop = self.pop if self.opt is None else Population.merge(self.opt, self.pop)
        self.opt = filter_optimum(pop, least_infeasible=True)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("es", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.ers = None


class SimpleCMAES(LocalSearch):
    def __init__(self, sigma=0.1, opts=None, normalize=True, **kwargs):
        super().__init__(**kwargs)
        self.termination = NoTermination()
        self.es = None
        self.sigma = sigma
        self.normalize = normalize
        self.norm = None

        DEFAULTS = {"verb_disp": 0}

        if opts is None:
            opts = {}

        for k, v in DEFAULTS.items():
            if k not in kwargs:
                opts[k] = v

        self.opts = opts

    def _setup(self, problem, **kwargs):
        xl, xu = problem.bounds()
        if self.normalize:
            self.norm, self.opts["bounds"] = bounds_if_normalize(xl, xu)
        else:
            self.norm = NoNormalization()
            self.opts["bounds"] = [xl, xu]
        self.opts["seed"] = self.seed

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills, **kwargs)
        x = self.norm.forward(self.x0.X)
        self.es = cma.CMAEvolutionStrategy(x, self.sigma, inopts=self.opts)

    def _infill(self):
        X = self.norm.backward(np.array(self.es.ask()))
        return Population.new("X", X)

    def _advance(self, infills=None, **kwargs):
        X, F = infills.get("X", "F")
        X = self.norm.forward(X)

        self.es.tell(X, F[:, 0])
        self.pop = infills

        if self.es.stop():
            self.termination.force_termination = True

    def _set_optimum(self):
        pop = self.pop if self.opt is None else Population.merge(self.opt, self.pop)
        self.opt = filter_optimum(pop, least_infeasible=True)


class BIPOPCMAES(CMAES):
    def __init__(self, restarts=4, **kwargs):
        super().__init__(restarts=restarts, bipop=True, **kwargs)


def bounds_if_normalize(xl, xu):
    norm = ZeroToOneNormalization(xl=xl, xu=xu)

    _xl, _xu = np.zeros_like(xl), np.ones_like(xu)
    if xl is not None:
        _xl[np.isnan(xl)] = np.nan
    if xu is not None:
        _xu[np.isnan(xu)] = np.nan

    return norm, [_xl, _xu]


parse_doc_string(CMAES.__init__)
