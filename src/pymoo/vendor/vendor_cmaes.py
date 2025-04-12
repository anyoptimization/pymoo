import time

import numpy as np
from cma import CMAOptions
from cma import optimization_tools as ot
from cma.evolution_strategy import cma_default_options, CMAEvolutionStrategy
from cma.utilities import utils
from cma.utilities.math import Mh

all_stoppings = []

def void(_):
    pass

def my_fmin(x0,
            sigma0,
            objective_function=void,
            options=None,
            args=(),
            gradf=None,
            restarts=0,
            restart_from_best='False',
            incpopsize=2,
            eval_initial_x=False,
            parallel_objective=None,
            noise_handler=None,
            noise_change_sigma_exponent=1,
            noise_kappa_exponent=0,  # TODO: add max kappa value as parameter
            bipop=False,
            callback=None):

    if 1 < 3:  # try: # pass on KeyboardInterrupt
        if not objective_function and not parallel_objective:  # cma.fmin(0, 0, 0)
            return CMAOptions()  # these opts are by definition valid

        fmin_options = locals().copy()  # archive original options
        del fmin_options['objective_function']
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
                    popsize_multiplier = fmin_options['incpopsize'] ** (irun - runs_with_small)
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

                sigma_factor = 0.01 ** np.random.uniform()  # Local search
                popsize_multiplier = fmin_options['incpopsize'] ** (irun - runs_with_small)
                opts['popsize'] = np.floor(popsize0 * popsize_multiplier ** (np.random.uniform() ** 2))
                opts['maxiter'] = min(maxiter0, 0.5 * sum(large_i) / opts['popsize'])
                # print('small basemul %s --> %s; maxiter %s' % (popsize_multiplier, opts['popsize'], opts['maxiter']))

            else:
                # A run with large population size; the population
                # doubling is implicit with incpopsize.
                poptype = 'large'

                popsize_multiplier = fmin_options['incpopsize'] ** (irun - runs_with_small)
                opts['popsize'] = popsize0 * popsize_multiplier
                opts['maxiter'] = maxiter0
                # print('large basemul %s --> %s; maxiter %s' % (popsize_multiplier, opts['popsize'], opts['maxiter']))

            if not callable(objective_function) and callable(parallel_objective):
                def objective_function(x, *args):
                    """created from `parallel_objective` argument"""
                    return parallel_objective([x], *args)[0]

            # recover from a CMA object
            if irun == 0 and isinstance(x0, MyCMAEvolutionStrategy):
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
                if callable(objective_function) and (
                        eval_initial_x
                        or es.opts['CMA_elitist'] == 'initial'
                        or (es.opts['CMA_elitist'] and
                            eval_initial_x is None)):
                    x = es.gp.pheno(es.mean,
                                    into_bounds=es.boundary_handler.repair,
                                    archive=es.sent_solutions)
                    es.f0 = yield x
                    es.best.update([x], es.sent_solutions,
                                   [es.f0], 1)
                    es.countevals += 1
            es.objective_function = objective_function  # only for the record

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
            try:
                logger.persistent_communication_dict.update(
                    {'variable_annotations':
                         objective_function.variable_annotations})
            except AttributeError:
                pass

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
                    X, fit = yield from es.ask_and_eval(parallel_objective or objective_function,
                                             args, gradf=gradf,
                                             evaluations=noisehandler.evaluations,
                                             aggregation=np.median,
                                             parallel_mode=parallel_objective)  # treats NaN with resampling if not parallel_mode
                    # TODO: check args and in case use args=(noisehandler.evaluations, )

                    if 11 < 3 and opts['vv']:  # inject a solution
                        # use option check_point = [0]
                        if 0 * np.random.randn() >= 0:
                            X[0] = 0 + opts['vv'] * es.sigma ** 0 * np.random.randn(es.N)
                            fit[0] = yield X[0]
                            # print fit[0]
                    if es.opts['verbose'] > 4:  # may be undesirable with dynamic fitness (e.g. Augmented Lagrangian)
                        if es.countiter < 2 or min(fit) <= es.best.last.f:
                            degrading_iterations_count = 0  # comes first to avoid code check complaint
                        else:  # min(fit) > es.best.last.f:
                            degrading_iterations_count += 1
                            if degrading_iterations_count > 4:
                                utils.print_message('%d f-degrading iterations (set verbose<=4 to suppress)'
                                                    % degrading_iterations_count,
                                                    iteration=es.countiter)
                    es.tell(X, fit)  # prepare for next iteration
                    if noise_handling:  # it would be better to also use these f-evaluations in tell
                        es.sigma *= noisehandler(X, fit, objective_function, es.ask,
                                                 args=args) ** fmin_opts['noise_change_sigma_exponent']

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
                    logger.add(
                        # more_data=[noisehandler.evaluations, 10**noisehandler.noiseS] if noise_handling else [],
                        modulo=1 if es.stop() and logger.modulo else None)
                    if (opts['verb_log'] and opts['verb_plot'] and
                            (es.countiter % max(opts['verb_plot'], opts['verb_log']) == 0 or es.stop())):
                        logger.plot(324)

            # end while not es.stop
            if opts['eval_final_mean'] and callable(objective_function):
                mean_pheno = es.gp.pheno(es.mean,
                                         into_bounds=es.boundary_handler.repair,
                                         archive=es.sent_solutions)
                fmean = yield mean_pheno
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
        if eval(safe_str(options['verb_disp'])) > 0:
            print(' in/outcomment ``raise`` in last line of cma.fmin to prevent/restore KeyboardInterrupt exception')
        raise KeyboardInterrupt  # cave: swallowing this exception can silently mess up experiments, if ctrl-C is hit


class MyCMAEvolutionStrategy(CMAEvolutionStrategy):

    def ask_and_eval(self, func, args=(), gradf=None, number=None, xmean=None, sigma_fac=1,
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
        X_first = self.ask(popsize, xmean=xmean, gradf=gradf, args=args)
        if xmean is None:
            xmean = self.mean  # might have changed in self.ask
        X = []
        if parallel_mode:
            if hasattr(func, 'evaluations'):
                evals0 = func.evaluations
            fit_first = yield X_first
            # the rest is only book keeping and warnings spitting
            if hasattr(func, 'evaluations'):
                self.countevals += func.evaluations - evals0 - self.popsize  # why not .sp.popsize ?
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
                        f = yield x
                    else:
                        f = yield xmean + kappa * length_normalizer * (x - xmean)

                if is_feasible(x, f) and evaluations > 1:
                    if kappa == 1:
                        _f = yield x
                        f = aggregation([f] + [_f])
                    else:
                        _f = []
                        for _i in range(int(evaluations - 1)):
                            v = yield xmean + kappa * length_normalizer * (x - xmean)
                            _f.append(v)
                        f = aggregation([f] + _f)

                if (rejected + 1) % 1000 == 0:
                    utils.print_warning('  %d solutions rejected (f-value NaN or None) at iteration %d' %
                          (rejected, self.countiter))
            fit.append(f)
            X.append(x)
        self.evaluations_per_f_value = int(evaluations)
        if any(f is None or utils.is_nan(f) for f in fit):
            idxs = [i for i in range(len(fit))
                    if fit[i] is None or utils.is_nan(fit[i])]
            utils.print_warning("f-values %s contain None or NaN at indices %s"
                                % (str(fit[:30]) + ('...' if len(fit) > 30 else ''),
                                   str(idxs)),
                                'ask_and_tell',
                                'CMAEvolutionStrategy',
                                self.countiter)
        return X, fit