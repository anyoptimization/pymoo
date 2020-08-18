import numpy as np

from pymoo.algorithms.so_local_search import LocalSearch
from pymoo.docs import parse_doc_string
from pymoo.model.algorithm import filter_optimum
from pymoo.model.population import Population
from pymoo.util.display import Display
from pymoo.util.termination.max_eval import MaximumFunctionCallTermination
from pymoo.util.termination.max_gen import MaximumGenerationTermination
from pymoo.util.termination.no_termination import NoTermination
from pymoo.vendor.vendor_cmaes import my_fmin


# =========================================================================================================
# Implementation
# =========================================================================================================


class CMAESDisplay(Display):

    def _do(self, problem, evaluator, algorithm):

        if algorithm.es.gi_frame is None:
            return

        super()._do(problem, evaluator, algorithm)

        fmin = algorithm.es.gi_frame.f_locals
        cma = fmin["es"]

        self.output.append("fopt", algorithm.opt[0].F[0])

        if fmin["restarts"] > 0:
            self.output.append("run", int(fmin["irun"]) + 1, width=4)
            self.output.append("fpop", algorithm.pop.get("F").min())
            self.output.append("n_pop", cma.opts['popsize'], width=5)

        self.output.append("sigma", cma.sigma)

        val = cma.sigma_vec * cma.dC ** 0.5
        self.output.append("min std", (cma.sigma * min(val)), width=8)
        self.output.append("max std", (cma.sigma * max(val)), width=8)

        axis = (cma.D.max() / cma.D.min()
                if not cma.opts['CMA_diagonal'] or cma.countiter > cma.opts['CMA_diagonal']
                else max(cma.sigma_vec * 1) / min(cma.sigma_vec * 1))
        self.output.append("axis", axis, width=8)


class CMAES(LocalSearch):

    def __init__(self,
                 x0=None,
                 sigma=0.5,
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
                 display=CMAESDisplay(),
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
              is (ab)used as scaling factor kappa (experimental).

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
              
        randn : np.random.randn  
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
        super().__init__(x0=x0, display=display, **kwargs)

        self.es = None
        self.cma = None

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

        self.default_termination = NoTermination()
        self.send_array_to_yield = True
        self.parallelize = parallelize
        self.al = None

    def setup(self, problem, seed=None, **kwargs):
        super().setup(problem, **kwargs)
        self.n_gen = 0

        xl = problem.xl.tolist() if problem.xl is not None else None
        xu = problem.xu.tolist() if problem.xu is not None else None

        self.options['bounds'] = [xl, xu]
        self.options['seed'] = seed

        if isinstance(self.termination, MaximumGenerationTermination):
            self.options['maxiter'] = self.termination.n_max_gen
        elif isinstance(self.termination, MaximumFunctionCallTermination):
            self.options['maxfevals'] = self.termination.n_max_evals

        # if self.problem.n_constr > 0:
        #     _al = AugmentedLagrangian(problem.n_var)
        #     _al.set_m(problem.n_constr)
        #     _al._equality = np.full(problem.n_constr, False)
        #     self.al = _al
        #     kwargs.setdefault('options', {}).setdefault('tolstagnation', 0)

    def _initialize(self):
        super()._initialize()
        self.pop = Population()

        kwargs = dict(
            options=self.options,
            parallelize=self.parallelize,
            restarts=self.restarts,
            restart_from_best=self.restart_from_best,
            incpopsize=self.incpopsize,
            eval_initial_x=self.eval_initial_x,
            noise_handler=self.noise_handler,
            noise_change_sigma_exponent=self.noise_change_sigma_exponent,
            noise_kappa_exponent=self.noise_kappa_exponent,
            bipop=self.bipop)

        self.es = my_fmin(self.x0.X, self.sigma, **kwargs)
        self._next()

    def _next(self):

        if self.pop is None or len(self.pop) == 0:
            X = next(self.es)

        else:
            F = self.pop.get("F")[:, 0].tolist()
            #
            # if self.problem.n_constr > 0:
            #     G = self.pop.get("G").tolist()
            #     self.al.set_coefficients(F, G)
            #
            #     x = self.es.gi_frame.f_locals["es"].ask(1, sigma_fac=0)[0]
            #     ind = Individual(X=x)
            #     self.evaluator.eval(self.problem, ind, algorithm=self)
            #     self.al.update(ind.F[0], ind.G)
            #
            #     F = F + sum(self.al(G))

            if not self.send_array_to_yield:
                F = F[0]

            try:
                X = self.es.send(F)
            except StopIteration:
                X = None
                self.termination.force_termination = True

        if X is not None:
            X = np.array(X)
            self.send_array_to_yield = X.ndim > 1
            X = np.atleast_2d(X)

            # evaluate the population
            self.pop = Population.new("X", X)
            self.evaluator.eval(self.problem, self.pop, algorithm=self)

            # set infeasible individual's objective values to np.nan - then CMAES can handle it
            for ind in self.pop:
                if not ind.feasible[0]:
                    ind.F[0] = np.nan

    def _set_optimum(self):
        val = self.pop
        if self.opt is not None:
            val = Population.merge(val, self.opt)
        self.opt = filter_optimum(val, least_infeasible=True)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["es"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.ers = None


parse_doc_string(CMAES.__init__)
