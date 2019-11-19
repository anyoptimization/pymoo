import numpy as np
from cma import CMAOptions
from cma import optimization_tools as ot
from cma.evolution_strategy import cma_default_options, CMAEvolutionStrategy
from cma.utilities import utils
from cma.utilities.math import Mh
import time

all_stoppings = []  # accessable via cma.evolution_strategy.all_stoppings, bound to change
def my_fmin(
         x0, 
         sigma0,
         options=None,
         args=(),
         gradf=None,
         restarts=0,
         restart_from_best='False',
         incpopsize=2,
         eval_initial_x=False,
         noise_handler=None,
         parallelize=True,
         noise_change_sigma_exponent=1,
         noise_kappa_exponent=0,  # TODO: add max kappa value as parameter
         bipop=False,
         callback=None):
    """functional interface to the stochastic optimizer CMA-ES
    for non-convex function minimization.

    Calling Sequences
    =================
    ``fmin(objective_function, x0, sigma0)``
        minimizes ``objective_function`` starting at ``x0`` and with
        standard deviation ``sigma0`` (step-size)
    ``fmin(objective_function, x0, sigma0, options={'ftarget': 1e-5})``
        minimizes ``objective_function`` up to target function value 1e-5,
        which is typically useful for benchmarking.
    ``fmin(objective_function, x0, sigma0, args=('f',))``
        minimizes ``objective_function`` called with an additional
        argument ``'f'``.
    ``fmin(objective_function, x0, sigma0, options={'ftarget':1e-5, 'popsize':40})``
        uses additional options ``ftarget`` and ``popsize``
    ``fmin(objective_function, esobj, None, options={'maxfevals': 1e5})``
        uses the `CMAEvolutionStrategy` object instance ``esobj`` to
        optimize ``objective_function``, similar to ``esobj.optimize()``.

    Arguments
    =========
    ``objective_function``
        called as ``objective_function(x, *args)`` to be minimized.
        ``x`` is a one-dimensional `numpy.ndarray`. See also the
        `parallel_objective` argument.
        ``objective_function`` can return `numpy.NaN`, which is
        interpreted as outright rejection of solution ``x`` and invokes
        an immediate resampling and (re-)evaluation of a new solution
        not counting as function evaluation. The attribute
        ``variable_annotations`` is passed into the
        ``CMADataLogger.persistent_communication_dict``.
    ``x0``
        list or `numpy.ndarray`, initial guess of minimum solution
        before the application of the geno-phenotype transformation
        according to the ``transformation`` option.  It can also be
        a string holding a Python expression that is evaluated
        to yield the initial guess - this is important in case
        restarts are performed so that they start from different
        places.  Otherwise ``x0`` can also be a `cma.CMAEvolutionStrategy`
        object instance, in that case ``sigma0`` can be ``None``.
    ``sigma0``
        scalar, initial standard deviation in each coordinate.
        ``sigma0`` should be about 1/4th of the search domain width
        (where the optimum is to be expected). The variables in
        ``objective_function`` should be scaled such that they
        presumably have similar sensitivity.
        See also `ScaleCoordinates`.
    ``options``
        a dictionary with additional options passed to the constructor
        of class ``CMAEvolutionStrategy``, see ``cma.CMAOptions`` ()
        for a list of available options.
    ``args=()``
        arguments to be used to call the ``objective_function``
    ``gradf=None``
        gradient of f, where ``len(gradf(x, *args)) == len(x)``.
        ``gradf`` is called once in each iteration if
        ``gradf is not None``.
    ``restarts=0``
        number of restarts with increasing population size, see also
        parameter ``incpopsize``, implementing the IPOP-CMA-ES restart
        strategy, see also parameter ``bipop``; to restart from
        different points (recommended), pass ``x0`` as a string.
    ``restart_from_best=False``
        which point to restart from
    ``incpopsize=2``
        multiplier for increasing the population size ``popsize`` before
        each restart
    ``parallel_objective``
        an objective function that accepts a list of `numpy.ndarray` as
        input and returns a `list`, which is mostly used instead of
        `objective_function`, but for the initial (also initial
        elitist) and the final evaluations. If ``parallel_objective``
        is given, the ``objective_function`` (first argument) may be
        ``None``.
    ``eval_initial_x=None``
        evaluate initial solution, for ``None`` only with elitist option
    ``noise_handler=None``
        a ``NoiseHandler`` class or instance or ``None``. Example:
        ``cma.fmin(f, 6 * [1], 1, noise_handler=cma.NoiseHandler(6))``
        see ``help(cma.NoiseHandler)``.
    ``noise_change_sigma_exponent=1``
        exponent for the sigma increment provided by the noise handler for
        additional noise treatment. 0 means no sigma change.
    ``noise_evaluations_as_kappa=0``
        instead of applying reevaluations, the "number of evaluations"
        is (ab)used as scaling factor kappa (experimental).
    ``bipop=False``
        if `True`, run as BIPOP-CMA-ES; BIPOP is a special restart
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
    ``callback=None``
        `callable` or list of callables called at the end of each
        iteration with the current `CMAEvolutionStrategy` instance
        as argument.

    Optional Arguments
    ==================
    All values in the `options` dictionary are evaluated if they are of
    type `str`, besides `verb_filenameprefix`, see class `CMAOptions` for
    details. The full list is available by calling ``cma.CMAOptions()``.

    >>> import cma
    >>> cma.CMAOptions()  #doctest: +ELLIPSIS
    {...

    Subsets of options can be displayed, for example like
    ``cma.CMAOptions('tol')``, or ``cma.CMAOptions('bound')``,
    see also class `CMAOptions`.

    Return
    ======
    Return the list provided in `CMAEvolutionStrategy.result` appended
    with termination conditions, an `OOOptimizer` and a `BaseDataLogger`::

        res = es.result + (es.stop(), es, logger)

    where
        - ``res[0]`` (``xopt``) -- best evaluated solution
        - ``res[1]`` (``fopt``) -- respective function value
        - ``res[2]`` (``evalsopt``) -- respective number of function evaluations
        - ``res[3]`` (``evals``) -- number of overall conducted objective function evaluations
        - ``res[4]`` (``iterations``) -- number of overall conducted iterations
        - ``res[5]`` (``xmean``) -- mean of the final sample distribution
        - ``res[6]`` (``stds``) -- effective stds of the final sample distribution
        - ``res[-3]`` (``stop``) -- termination condition(s) in a dictionary
        - ``res[-2]`` (``cmaes``) -- class `CMAEvolutionStrategy` instance
        - ``res[-1]`` (``logger``) -- class `CMADataLogger` instance

    Details
    =======
    This function is an interface to the class `CMAEvolutionStrategy`. The
    latter class should be used when full control over the iteration loop
    of the optimizer is desired.

    Examples
    ========
    The following example calls `fmin` optimizing the Rosenbrock function
    in 10-D with initial solution 0.1 and initial step-size 0.5. The
    options are specified for the usage with the `doctest` module.

    >>> import cma
    >>> # cma.CMAOptions()  # returns all possible options
    >>> options = {'CMA_diagonal':100, 'seed':1234, 'verb_time':0}
    >>>
    >>> res = cma.fmin(cma.ff.rosen, [0.1] * 10, 0.3, options)  #doctest: +ELLIPSIS
    (5_w,10)-aCMA-ES (mu_w=3.2,w_1=45%) in dimension 10 (seed=1234...)
       Covariance matrix is diagonal for 100 iterations (1/ccov=26...
    Iterat #Fevals   function value  axis ratio  sigma ...
        1     10 ...
    termination on tolfun=1e-11 ...
    final/bestever f-value = ...
    >>> assert res[1] < 1e-12  # f-value of best found solution
    >>> assert res[2] < 8000  # evaluations

    The above call is pretty much equivalent with the slightly more
    verbose call::

        res = cma.CMAEvolutionStrategy([0.1] * 10, 0.3,
                    options=options).optimize(cma.ff.rosen).result

    where `optimize` returns a `CMAEvolutionStrategy` instance. The
    following example calls `fmin` optimizing the Rastrigin function
    in 3-D with random initial solution in [-2,2], initial step-size 0.5
    and the BIPOP restart strategy (that progressively increases population).
    The options are specified for the usage with the `doctest` module.

    >>> import cma
    >>> # cma.CMAOptions()  # returns all possible options
    >>> options = {'seed':12345, 'verb_time':0, 'ftarget': 1e-8}
    >>>
    >>> res = cma.fmin(cma.ff.rastrigin, '2. * np.random.rand(3) - 1', 0.5,
    ...                options, restarts=9, bipop=True)  #doctest: +ELLIPSIS
    (3_w,7)-aCMA-ES (mu_w=2.3,w_1=58%) in dimension 3 (seed=12345...

    In either case, the method::

        cma.plot();

    (based on `matplotlib.pyplot`) produces a plot of the run and, if
    necessary::

        cma.s.figshow()

    shows the plot in a window. Finally::

        cma.s.figsave('myfirstrun')  # figsave from matplotlib.pyplot

    will save the figure in a png.

    We can use the gradient like

    >>> import cma
    >>> res = cma.fmin(cma.ff.rosen, np.zeros(10), 0.1,
    ...             options = {'ftarget':1e-8,},
    ...             gradf=cma.ff.grad_rosen,
    ...         )  #doctest: +ELLIPSIS
    (5_w,...
    >>> assert cma.ff.rosen(res[0]) < 1e-8
    >>> assert res[2] < 3600  # 1% are > 3300
    >>> assert res[3] < 3600  # 1% are > 3300

    If solution can only be comparatively ranked, either use
    `CMAEvolutionStrategy` directly or the objective accepts a list
    of solutions as input:

    >>> def parallel_sphere(X): return [cma.ff.sphere(x) for x in X]
    >>> x, es = cma.fmin2(None, 3 * [0], 0.1, {'verbose': -9},
    ...                   parallel_objective=parallel_sphere)
    >>> assert es.result[1] < 1e-9

    :See also: `CMAEvolutionStrategy`, `OOOptimizer.optimize`, `plot`,
        `CMAOptions`, `scipy.optimize.fmin`

    """  # style guides say there should be the above empty line

    if 1 < 3:  # try: # pass on KeyboardInterrupt

        fmin_options = locals().copy()  # archive original options
        del fmin_options['x0']
        del fmin_options['sigma0']
        del fmin_options['options']
        del fmin_options['args']

        if options is None:
            options = cma_default_options
        CMAOptions().check_attributes(options)  # might modify options
        # checked that no options.ftarget =
        opts = CMAOptions(options.copy()).complement()

        if callback is None:
            callback = []
        elif callable(callback):
            callback = [callback]

        # BIPOP-related variables:
        runs_with_small = 0
        small_i = []
        large_i = []
        popsize0 = None  # to be evaluated after the first iteration
        maxiter0 = None  # to be evaluated after the first iteration
        base_evals = 0

        irun = 0
        best = ot.BestSolution()
        all_stoppings = []
        while True:  # restart loop
            sigma_factor = 1

            # Adjust the population according to BIPOP after a restart.
            if not bipop:
                # BIPOP not in use, simply double the previous population
                # on restart.
                if irun > 0:
                    popsize_multiplier = fmin_options['incpopsize']**(irun - runs_with_small)
                    opts['popsize'] = popsize0 * popsize_multiplier

            elif irun == 0:
                # Initial run is with "normal" population size; it is
                # the large population before first doubling, but its
                # budget accounting is the same as in case of small
                # population.
                poptype = 'small'

            elif sum(small_i) < sum(large_i):
                # An interweaved run with small population size
                poptype = 'small'
                if 11 < 3:  # not needed when compared to irun - runs_with_small
                    restarts += 1  # A small restart doesn't count in the total
                runs_with_small += 1  # _Before_ it's used in popsize_lastlarge

                sigma_factor = 0.01**np.random.uniform()  # Local search
                popsize_multiplier = fmin_options['incpopsize']**(irun - runs_with_small)
                opts['popsize'] = np.floor(popsize0 * popsize_multiplier**(np.random.uniform()**2))
                opts['maxiter'] = min(maxiter0, 0.5 * sum(large_i) / opts['popsize'])
                # print('small basemul %s --> %s; maxiter %s' % (popsize_multiplier, opts['popsize'], opts['maxiter']))

            else:
                # A run with large population size; the population
                # doubling is implicit with incpopsize.
                poptype = 'large'

                popsize_multiplier = fmin_options['incpopsize']**(irun - runs_with_small)
                opts['popsize'] = popsize0 * popsize_multiplier
                opts['maxiter'] = maxiter0
                # print('large basemul %s --> %s; maxiter %s' % (popsize_multiplier, opts['popsize'], opts['maxiter']))

            # recover from a CMA object
            if irun == 0 and isinstance(x0, CMAEvolutionStrategy):
                es = x0
                x0 = es.inputargs['x0']  # for the next restarts
                if np.isscalar(sigma0) and np.isfinite(sigma0) and sigma0 > 0:
                    es.sigma = sigma0
                # debatable whether this makes sense:
                sigma0 = es.inputargs['sigma0']  # for the next restarts
                if options is not None:
                    es.opts.set(options)
                # ignore further input args and keep original options
            else:  # default case
                if irun and eval(str(fmin_options['restart_from_best'])):
                    utils.print_warning('CAVE: restart_from_best is often not useful',
                                        verbose=opts['verbose'])
                    es = MyCMAEvolutionStrategy(best.x, sigma_factor * sigma0, opts)
                else:
                    es = MyCMAEvolutionStrategy(x0, sigma_factor * sigma0, opts)
                # return opts, es
                if (eval_initial_x
                        or es.opts['CMA_elitist'] == 'initial'
                        or (es.opts['CMA_elitist'] and
                                    eval_initial_x is None)):
                    x = es.gp.pheno(es.mean,
                                    into_bounds=es.boundary_handler.repair,
                                    archive=es.sent_solutions)
                    es.f0 = yield es, x
                    es.best.update([x], es.sent_solutions,
                                   [es.f0], 1)
                    es.countevals += 1

            opts = es.opts  # processed options, unambiguous
            # a hack:
            fmin_opts = CMAOptions("unchecked", **fmin_options.copy())
            for k in fmin_opts:
                # locals() cannot be modified directly, exec won't work
                # in 3.x, therefore
                fmin_opts.eval(k, loc={'N': es.N,
                                       'popsize': opts['popsize']},
                               correct_key=False)

            es.logger.append = opts['verb_append'] or es.countiter > 0 or irun > 0
            # es.logger is "the same" logger, because the "identity"
            # is only determined by the `verb_filenameprefix` option
            logger = es.logger  # shortcut

            if 11 < 3:
                if es.countiter == 0 and es.opts['verb_log'] > 0 and \
                        not es.opts['verb_append']:
                   logger = CMADataLogger(es.opts['verb_filenameprefix']
                                            ).register(es)
                   logger.add()
                es.writeOutput()  # initial values for sigma etc

            if noise_handler:
                if isinstance(noise_handler, type):
                    noisehandler = noise_handler(es.N)
                else:
                    noisehandler = noise_handler
                noise_handling = True
                if fmin_opts['noise_change_sigma_exponent'] > 0:
                    es.opts['tolfacupx'] = inf
            else:
                noisehandler = ot.NoiseHandler(es.N, 0)  # switched off
                noise_handling = False
            es.noise_handler = noisehandler

            # the problem: this assumes that good solutions cannot take longer than bad ones:
            # with EvalInParallel(objective_function, 2, is_feasible=opts['is_feasible']) as eval_in_parallel:
            if 1 < 3:
                while not es.stop():  # iteration loop
                    # X, fit = eval_in_parallel(lambda: es.ask(1)[0], es.popsize, args, repetitions=noisehandler.evaluations-1)
                    X, fit = yield from es.ask_and_eval(gradf=gradf,
                                             evaluations=noisehandler.evaluations,
                                             aggregation=np.median,
                                             parallel_mode=parallelize)  # treats NaN with resampling if not parallel_mode

                    if 11 < 3 and opts['vv']:  # inject a solution
                        # use option check_point = [0]
                        if 0 * np.random.randn() >= 0:
                            X[0] = 0 + opts['vv'] * es.sigma**0 * np.random.randn(es.N)
                            fit[0] = yield es, X[0]
                            # print fit[0]
                    if es.opts['verbose'] > 4:
                        if es.countiter > 1 and min(fit) > es.best.last.f:
                            unsuccessful_iterations_count += 1
                            if unsuccessful_iterations_count > 4:
                                utils.print_message('%d unsuccessful iterations'
                                                    % unsuccessful_iterations_count,
                                                    iteration=es.countiter)
                        else:
                            unsuccessful_iterations_count = 0
                    es.tell(X, fit)  # prepare for next iteration
                    if noise_handling:  # it would be better to also use these f-evaluations in tell
                        es.sigma *= noisehandler(X, fit, objective_function, es.ask,
                                                 args=args)**fmin_opts['noise_change_sigma_exponent']

                        es.countevals += noisehandler.evaluations_just_done  # TODO: this is a hack, not important though
                        # es.more_to_write.append(noisehandler.evaluations_just_done)
                        if noisehandler.maxevals > noisehandler.minevals:
                            es.more_to_write.append(noisehandler.evaluations)
                        if 1 < 3:
                            # If sigma was above multiplied by the same
                            #  factor cmean is divided by here, this is
                            #  like only multiplying kappa instead of
                            #  changing cmean and sigma.
                            es.sp.cmean *= np.exp(-noise_kappa_exponent * np.tanh(noisehandler.noiseS))
                            es.sp.cmean[es.sp.cmean > 1] = 1.0  # also works with "scalar arrays" like np.array(1.2)
                    for f in callback:
                        f is None or f(es)
                    es.disp()
                    logger.add(# more_data=[noisehandler.evaluations, 10**noisehandler.noiseS] if noise_handling else [],
                               modulo=1 if es.stop() and logger.modulo else None)
                    if (opts['verb_log'] and opts['verb_plot'] and
                          (es.countiter % max(opts['verb_plot'], opts['verb_log']) == 0 or es.stop())):
                        logger.plot(324)

            # end while not es.stop
            if opts['eval_final_mean']:
                mean_pheno = es.gp.pheno(es.mean,
                                         into_bounds=es.boundary_handler.repair,
                                         archive=es.sent_solutions)
                fmean = yield es, mean_pheno
                es.countevals += 1
                es.best.update([mean_pheno], es.sent_solutions, [fmean], es.countevals)

            best.update(es.best, es.sent_solutions)  # in restarted case
            # es.best.update(best)

            this_evals = es.countevals - base_evals
            base_evals = es.countevals

            # BIPOP stats update

            if irun == 0:
                popsize0 = opts['popsize']
                maxiter0 = opts['maxiter']
                # XXX: This might be a bug? Reproduced from Matlab
                # small_i.append(this_evals)

            if bipop:
                if poptype == 'small':
                    small_i.append(this_evals)
                else:  # poptype == 'large'
                    large_i.append(this_evals)

            # final message
            if opts['verb_disp']:
                es.result_pretty(irun, time.asctime(time.localtime()),
                                 best.f)

            irun += 1
            # if irun > fmin_opts['restarts'] or 'ftarget' in es.stop() \
            # if irun > restarts or 'ftarget' in es.stop() \
            all_stoppings.append(dict(es.stop(check=False)))  # keeping the order
            if irun - runs_with_small > fmin_opts['restarts'] or 'ftarget' in es.stop() \
                    or 'maxfevals' in es.stop(check=False) or 'callback' in es.stop(check=False):
                break
            opts['verb_append'] = es.countevals
            opts['popsize'] = fmin_opts['incpopsize'] * es.sp.popsize  # TODO: use rather options?
            try:
                opts['seed'] += 1
            except TypeError:
                pass

        # while irun

        # es.out['best'] = best  # TODO: this is a rather suboptimal type for inspection in the shell
        if irun:
            es.best.update(best)
            # TODO: there should be a better way to communicate the overall best
        return es.result + (es.stop(), es, logger)
        ### 4560
        # TODO refine output, can #args be flexible?
        # is this well usable as it is now?
    else:  # except KeyboardInterrupt:  # Exception as e:
        if eval(str(options['verb_disp'])) > 0:
            print(' in/outcomment ``raise`` in last line of cma.fmin to prevent/restore KeyboardInterrupt exception')
        raise KeyboardInterrupt  # cave: swallowing this exception can silently mess up experiments, if ctrl-C is hit


class MyCMAEvolutionStrategy(CMAEvolutionStrategy):


    def ask_and_eval(self, gradf=None, number=None, xmean=None, sigma_fac=1,
                     evaluations=1, aggregation=np.median, kappa=1, parallel_mode=False):

        # initialize
        popsize = self.sp.popsize
        if number is not None:
            popsize = int(number)

        if self.opts['CMA_mirrormethod'] == 1:  # direct selective mirrors
            nmirrors = Mh.sround(self.sp.lam_mirr * popsize / self.sp.popsize)
            self._mirrormethod1_done = self.countiter
        else:
            # method==0 unconditional mirrors are done in ask_geno
            # method==2 delayed selective mirrors are done via injection
            nmirrors = 0
        assert nmirrors <= popsize // 2
        self.mirrors_idx = np.arange(nmirrors)  # might never be used
        is_feasible = self.opts['is_feasible']

        # do the work
        fit = []  # or np.NaN * np.empty(number)
        X_first = self.ask(popsize, xmean=xmean, gradf=gradf, args=[])
        if xmean is None:
            xmean = self.mean  # might have changed in self.ask
        X = []
        if parallel_mode:
            fit_first = yield self, X_first

            # the rest is only book keeping and warnings spitting
            """
            if hasattr(func, 'last_evaluations'):
                self.countevals += func.last_evaluations - self.popsize
            elif hasattr(func, 'evaluations'):
                if self.countevals < func.evaluations:
                    self.countevals = func.evaluations - self.popsize
            """

            if nmirrors and self.opts['CMA_mirrormethod'] > 0 and self.countiter < 2:
                utils.print_warning(
                    "selective mirrors will not work in parallel mode",
                    "ask_and_eval", "CMAEvolutionStrategy")
            if evaluations > 1 and self.countiter < 2:
                utils.print_warning(
                    "aggregating evaluations will not work in parallel mode",
                    "ask_and_eval", "CMAEvolutionStrategy")
        else:
            fit_first = len(X_first) * [None]
            
        for k in range(popsize):
            x, f = X_first.pop(0), fit_first.pop(0)
            rejected = -1
            while f is None or not is_feasible(x, f):  # rejection sampling
                if parallel_mode:
                    utils.print_warning(
                        "rejection sampling will not work in parallel mode"
                        " unless the parallel_objective makes a distinction\n"
                        "between called with a numpy array vs a list (of"
                        " numpy arrays) as first argument.",
                        "ask_and_eval", "CMAEvolutionStrategy")
                rejected += 1
                if rejected:  # resample
                    x = self.ask(1, xmean, sigma_fac)[0]
                elif k >= popsize - nmirrors:  # selective mirrors
                    if k == popsize - nmirrors:
                        self.mirrors_idx = np.argsort(fit)[-1:-1 - nmirrors:-1]
                    x = self.get_mirror(X[self.mirrors_idx[popsize - 1 - k]])

                # constraints handling test hardwired ccccccccccc

                length_normalizer = 1
                # zzzzzzzzzzzzzzzzzzzzzzzzz
                if 11 < 3:
                    # for some unclear reason, this normalization does not work as expected: the step-size
                    # becomes sometimes too large and overall the mean might diverge. Is the reason that
                    # we observe random fluctuations, because the length is not selection relevant?
                    # However sigma-adaptation should mainly work on the correlation, not the length?
                    # Or is the reason the deviation of the direction introduced by using the original
                    # length, which also can effect the measured correlation?
                    # Update: if the length of z in CSA is clipped at chiN+1, it works, but only sometimes?
                    length_normalizer = self.N**0.5 / self.mahalanobis_norm(x - xmean)  # self.const.chiN < N**0.5, the constant here is irrelevant (absorbed by kappa)
                    # print(self.N**0.5 / self.mahalanobis_norm(x - xmean))
                    # self.more_to_write += [length_normalizer * 1e-3, length_normalizer * self.mahalanobis_norm(x - xmean) * 1e2]

                if kappa == 1:
                    f = yield self, x
                else:
                    f = yield self, xmean + kappa * length_normalizer * (x - xmean)

                if is_feasible(x, f) and evaluations > 1:

                    _f = []
                    for _i in range(int(evaluations - 1)):
                        if kappa == 1:
                            __f = yield self, x
                        else:
                            __f = yield self, xmean + kappa * length_normalizer * (x - xmean)

                        _f.append(__f)

                    f = aggregation([f] + _f)
                if (rejected + 1) % 1000 == 0:
                    utils.print_warning('  %d solutions rejected (f-value NaN or None) at iteration %d' %
                          (rejected, self.countiter))
            fit.append(f)
            X.append(x)
        self.evaluations_per_f_value = int(evaluations)
        if any(f is None or np.isnan(f) for f in fit):
            idxs = [i for i in range(len(fit))
                    if fit[i] is None or np.isnan(fit[i])]
            utils.print_warning("f-values %s contain None or NaN at indices %s"
                                % (str(fit[:30]) + ('...' if len(fit) > 30 else ''),
                                   str(idxs)),
                                'ask_and_tell',
                                'CMAEvolutionStrategy',
                                self.countiter)
        return X, fit