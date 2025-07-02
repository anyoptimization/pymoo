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

        val = cma.sigma_vec * cma.dC ** 0.5
        self.min_std.set((cma.sigma * min(val)))
        self.max_std.set((cma.sigma * max(val)))

        if algorithm.restarts > 0:
            self.run.set(int(fmin["irun"] - fmin["runs_with_small"]) + 1)
            self.fpop.set(algorithm.pop.get("F").min())
            self.n_pop.set(int(cma.opts['popsize']))

        axis = (cma.D.max() / cma.D.min()
                if not cma.opts['CMA_diagonal'] or cma.countiter > cma.opts['CMA_diagonal']
                else max(cma.sigma_vec * 1) / min(cma.sigma_vec * 1))
        self.axis.set(axis)


class CMAES(LocalSearch):

    def __init__(self,
                 x0=None,
                 sigma=0.1,
                 normalize=True,
                 parallelize=True,
                 maxfevals=np.inf,
                 tolfun=1e-11,
                 tolx=1e-11,
                 restarts=0,
                 restart_from_best='False',
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
                 **kwargs
                 ):
        """


        Parameters
        ----------

        x0 : list or `numpy.ndarray`
              initial guess of minimum solution
              before the application of the geno-phenotype transformation
              according to the ``transformation`` option.  It can also be
              a string holding a Python expression that is evaluated
              to yield the initial guess - this is important in case
              restarts are performed so that they start from different
              places.  Otherwise ``x0`` can also be a `cma.CMAEvolutionStrategy`
              object instance, in that case ``sigma0`` can be ``None``.

        sigma : float
              Initial standard deviation in each coordinate.
              ``sigma0`` should be about 1/4th of the search domain width
              (where the optimum is to be expected). The variables in
              ``objective_function`` should be scaled such that they
              presumably have similar sensitivity.
              See also `ScaleCoordinates`.

        parallelize : bool
              Whether the objective function should be called for each single evaluation or batch wise.

        restarts : int, default 0
              Number of restarts with increasing population size, see also
              parameter ``incpopsize``, implementing the IPOP-CMA-ES restart
              strategy, see also parameter ``bipop``; to restart from
              different points (recommended), pass ``x0`` as a string.

        restart_from_best : bool, default false
               Which point to restart from

        incpopsize : int
              Multiplier for increasing the population size ``popsize`` before each restart

        eval_initial_x : bool
              Evaluate initial solution, for ``None`` only with elitist option

        noise_handler : class
              A ``NoiseHandler`` class or instance or ``None``. Example:
              ``cma.fmin(f, 6 * [1], 1, noise_handler=cma.NoiseHandler(6))``
              see ``help(cma.NoiseHandler)``.

        noise_change_sigma_exponent : int
              Exponent for the sigma increment provided by the noise handler for
              additional noise treatment. 0 means no sigma change.

        noise_kappa_exponent : int
              Instead of applying reevaluations, the "number of evaluations"
              is (ab)used as init_simplex_scale factor kappa (experimental).

        bipop : bool
              If `True`, run as BIPOP-CMA-ES; BIPOP is a special restart
              strategy switching between two population sizings - small
              (like the default CMA, but with more focused search) and
              large (progressively increased as in IPOP). This makes the
              algorithm perform well both on functions with many regularly
              or irregularly arranged local optima (the latter by frequently
              restarting with small populations).  For the `bipop` parameter
              to actually take effect, also select non-zero number of
              (IPOP) restarts; the recommended setting is ``restarts<=9``
              and `x0` passed as a string using `numpy.rand` to generate
              initial solutions. Note that small-population restarts
              do not count into the total restart count.

        AdaptSigma : True
              Or False or any CMAAdaptSigmaBase class e.g. CMAAdaptSigmaTPA, CMAAdaptSigmaCSA

        CMA_active : True
              Negative update, conducted after the original update

        CMA_activefac : 1
              Learning rate multiplier for active update

        CMA_cmean : 1
              Learning rate for the mean value

        CMA_const_trace : False
            Normalize trace, 1, True, "arithm", "geom", "aeig", "geig" are valid

        CMA_diagonal : 0*100*N/popsize**0.5
            Number of iterations with diagonal covariance matrix, True for always

        CMA_eigenmethod : np.linalg.eigh or cma.utilities.math.eig or pygsl.eigen.eigenvectors

        CMA_elitist : False  or "initial" or True
            Elitism likely impairs global search performance

        CMA_injections_threshold_keep_len : 0
            Keep length if Mahalanobis length is below the given relative threshold

        CMA_mirrors : popsize < 6
            Values <0.5 are interpreted as fraction, values >1 as numbers (rounded), otherwise about 0.16 is used

        CMA_mirrormethod : int, default 2,  0=unconditional, 1=selective, 2=selective with delay

        CMA_mu : None
            Parents selection parameter, default is popsize // 2

        CMA_on : 1
            Multiplier for all covariance matrix updates

        CMA_sampler : None
            A class or instance that implements the interface of
              `cma.interfaces.StatisticalModelSamplerWithZeroMeanBaseClass`

        CMA_sampler_options : dict
            Options passed to `CMA_sampler` class init as keyword arguments

        CMA_rankmu : 1.0
            Multiplier for rank-mu update learning rate of covariance matrix

        CMA_rankone : 1.0
            Multiplier for rank-one update learning rate of covariance matrix

        CMA_recombination_weights : None
            A list, see class RecombinationWeights, overwrites CMA_mu and popsize options

        CMA_dampsvec_fac : np.Inf
            Tentative and subject to changes, 0.5 would be a "default" damping for sigma vector update

        CMA_dampsvec_fade : 0.1
            Tentative fading out parameter for sigma vector update

        CMA_teststds : None
            Factors for non-isotropic initial distr. of C, mainly for test purpose, see CMA_stds for production

        CMA_stds : None
            Multipliers for sigma0 in each coordinate, not represented in C,
            makes scaling_of_variables obsolete

        CSA_dampfac : 1
            Positive multiplier for step-size damping, 0.3 is close to optimal on the sphere

        CSA_damp_mueff_exponent : 0.5
            Zero would mean no dependency of damping on mueff, useful with CSA_disregard_length option

        CSA_disregard_length : False
            True is untested, also changes respective parameters

        CSA_clip_length_value : None
            Poorly tested, [0, 0] means const length N**0.5, [-1, 1] allows a variation of +- N/(N+2), etc.

        CSA_squared : False
            Use squared length for sigma-adaptation ',

        BoundaryHandler : BoundTransform or BoundPenalty, unused when ``bounds in (None, [None, None])``

        conditioncov_alleviate : [1e8, 1e12]
            When to alleviate the condition in the coordinates and in main axes

        eval_final_mean : True
            Evaluate the final mean, which is a favorite return candidate

        fixed_variables : None
            Dictionary with index-value pairs like dict(0=1.1, 2=0.1) that are not optimized

        ftarget : -inf
            Target function value, minimization

        integer_variables : []
            Index list, invokes basic integer handling: prevent std dev to become too small in the given variables

        maxfevals : inf
            Maximum number of function evaluations

        maxiter : 100 + 150 * (N+3)**2 // popsize**0.5
            Maximum number of iterations

        mean_shift_line_samples : False
            Sample two new solutions colinear to previous mean shift

        mindx : 0
            Minimal std in any arbitrary direction, cave interference with tol

        minstd : 0
            Minimal std (scalar or vector) in any coordinate direction, cave interference with tol

        maxstd : inf
            Maximal std in any coordinate direction

        pc_line_samples : False
            One line sample along the evolution path pc

        popsize : 4+int(3*np.log(N))
            Population size, AKA lambda, number of new solution per iteration

        randn
            Randn(lam, N) must return an np.array of shape (lam, N), see also cma.utilities.math.randhss

        signals_filename : None
            cma_signals.in  # read versatile options from this file which contains a single options dict,
            e.g. ``dict("timeout"=0)`` to stop, string-values are evaluated, e.g. "np.inf" is valid

        termination_callback : None
            A function returning True for termination, called in `stop` with `self` as argument, could be abused
            for side effects

        timeout : inf
            Stop if timeout seconds are exceeded, the string "2.5 * 60**2" evaluates to 2 hours and 30 minutes

        tolconditioncov : 1e14
            Stop if the condition of the covariance matrix is above `tolconditioncov`

        tolfacupx : 1e3
            Termination when step-size increases by tolfacupx (diverges). That is, the initial step-size was chosen
            far too small and better solutions were found far away from the initial solution x0

        tolupsigma : 1e20
            Sigma/sigma0 > tolupsigma * max(eivenvals(C)**0.5) indicates "creeping behavior" with usually minor
            improvements

        tolfun : 1e-11
            Termination criterion: tolerance in function value, quite useful

        tolfunhist : 1e-12
            Termination criterion: tolerance in function value history

        tolstagnation : int(100 + 100 * N**1.5 / popsize)
            Termination if no improvement over tolstagnation iterations

        tolx : 1e-11
            Termination criterion: tolerance in x-changes

        typical_x : None
            Used with scaling_of_variables',

        updatecovwait : None
            Number of iterations without distribution update, name is subject to future changes

        cmaes_verbose : 3
            Verbosity e.g. of initial/final message, -1 is very quiet, -9 maximally quiet, may not be fully implemented

        verb_append : 0
            Initial evaluation counter, if append, do not overwrite output files

        verb_disp : 100
            Verbosity: display console output every verb_disp iteration

        verb_filenameprefix : str
            CMADataLogger.default_prefix + Output path and filenames prefix

        verb_log : 1
            Verbosity: write data to files every verb_log iteration, writing can be time critical on fast to
            evaluate functions

        verb_plot : 0
              In fmin(): plot() is called every verb_plot iteration

        verb_time : True
              Output timings on console

        vv : dict
            Versatile set or dictionary for hacking purposes, value found in self.opts["vv"]

        kwargs : dict
              A dictionary with additional options passed to the constructor
              of class ``CMAEvolutionStrategy``, see ``cma.CMAOptions`` ()
              for a list of available options.

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
            **kwargs
        )

        self.send_array_to_yield = True
        self.parallelize = parallelize
        self.al = None

    def _setup(self, problem, **kwargs):

        xl, xu = problem.bounds()
        if self.normalize:
            self.norm, self.options['bounds'] = bounds_if_normalize(xl, xu)
        else:
            self.norm = NoNormalization()
            self.options['bounds'] = [xl, xu]

        seed = kwargs.get('seed', self.seed)
        self.options['seed'] = seed

        if isinstance(self.termination, MaximumGenerationTermination):
            self.options['maxiter'] = self.termination.n_max_gen
        elif isinstance(self.termination, MaximumFunctionCallTermination):
            self.options['maxfevals'] = self.termination.n_max_evals

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
            random_state=self.random_state)

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
            except:
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
            self.norm, self.opts['bounds'] = bounds_if_normalize(xl, xu)
        else:
            self.norm = NoNormalization()
            self.opts['bounds'] = [xl, xu]
        self.opts['seed'] = self.seed

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
