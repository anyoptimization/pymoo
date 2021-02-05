import time

import numpy as np
from cma import CMAOptions
from cma import optimization_tools as ot
from cma.evolution_strategy import cma_default_options, CMAEvolutionStrategy
from cma.utilities import utils

from pymoo.algorithms.base.local import LocalSearch
from pymoo.docs import parse_doc_string
from pymoo.model.population import Population
from pymoo.util.display import Display
from pymoo.util.optimum import filter_optimum
from pymoo.util.termination.max_eval import MaximumFunctionCallTermination
from pymoo.util.termination.max_gen import MaximumGenerationTermination


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
            self.output.append("n_pop", int(cma.opts['popsize']), width=5)

        self.output.append("sigma", cma.sigma)

        val = cma.sigma_vec * cma.dC ** 0.5
        self.output.append("min std", (cma.sigma * min(val)), width=8)
        self.output.append("max std", (cma.sigma * max(val)), width=8)

        axis = (cma.D.max() / cma.D.min()
                if not cma.opts['CMA_diagonal'] or cma.countiter > cma.opts['CMA_diagonal']
                else max(cma.sigma_vec * 1) / min(cma.sigma_vec * 1))
        self.output.append("axis", axis, width=8)


def delete_if_in_dict(d, keys):
    for k in keys:
        if k in d:
            del d[k]
    return d


class CMAES(LocalSearch):

    def __init__(self,
                 sigma0=0.5,
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
                 noise_kappa_exponent=0,
                 bipop=False,
                 callback=None,
                 display=CMAESDisplay(),
                 **kwargs
                 ):

        super().__init__(display=display, **kwargs)

        rem_attrs = ['self', 'display', '__class__', 'x0', 'sigma0', 'options', 'args', 'rem_attrs']
        self.fmin_options = delete_if_in_dict(locals().copy(), rem_attrs)

        if options is None:
            options = cma_default_options

        CMAOptions().check_attributes(options)  # might modify options
        opts = CMAOptions(options.copy()).complement()
        self.options = options

        self.opts = None

        if callback is None:
            callback = []
        elif callable(callback):
            callback = [callback]
        self.callback = callback

        self.logger = None

        self.sigma0 = sigma0
        self.args = args
        self.gradf = gradf
        self.restarts = restarts
        self.restart_from_best = restart_from_best
        self.incpopsize = incpopsize
        self.eval_initial_x = eval_initial_x

        self.noise_handler = noise_handler
        self.noise_handling = None

        self.parallelize = parallelize
        self.noise_change_sigma_exponent = noise_change_sigma_exponent
        self.noise_kappa_exponent = noise_kappa_exponent
        self.bipop = bipop
        self.poptype = None

        self.unsuccessful_iterations_count = 0

        # BIPOP-related variables:
        self.runs_with_small = 0
        self.small_i = []
        self.large_i = []
        self.popsize0 = None  # to be evaluated after the first iteration
        self.maxiter0 = None  # to be evaluated after the first iteration
        self.popsize_multiplier = None
        self.base_evals = 0

        self.irun = 0
        self.best = ot.BestSolution()
        self.all_stoppings = []

    def _setup(self, problem, seed=None, **kwargs):
        super()._setup(problem, **kwargs)

        xl = problem.xl.tolist() if problem.xl is not None else None
        xu = problem.xu.tolist() if problem.xu is not None else None

        options = self.options
        options['bounds'] = [xl, xu]
        options['seed'] = seed

        if isinstance(self.termination, MaximumGenerationTermination):
            options['maxiter'] = self.termination.n_max_gen
        elif isinstance(self.termination, MaximumFunctionCallTermination):
            options['maxfevals'] = self.termination.n_max_evals

        self.opts = CMAOptions(options.copy()).complement()

    def _restart(self):
        opts, fmin_options = self.opts, self.fmin_options

        self.sigma_factor = 1

        # Adjust the population according to BIPOP after a restart.
        if not self.bipop:
            # BIPOP not in use, simply double the previous population
            # on restart.
            if self.irun > 0:
                self.popsize_multiplier = fmin_options['incpopsize'] ** (self.irun - self.runs_with_small)
                opts['popsize'] = self.popsize0 * self.popsize_multiplier

        elif self.irun == 0:
            # Initial run is with "normal" population size; it is
            # the large population before first doubling, but its
            # budget accounting is the same as in case of small
            # population.
            self.poptype = 'small'

        elif sum(self.small_i) < sum(self.large_i):
            # An interweaved run with small population size
            self.poptype = 'small'
            if 11 < 3:  # not needed when compared to irun - runs_with_small
                self.restarts += 1  # A small restart doesn't count in the total
            self.runs_with_small += 1  # _Before_ it's used in popsize_lastlarge

            self.sigma_factor = 0.01 ** np.random.uniform()  # Local search
            self.popsize_multiplier = fmin_options['incpopsize'] ** (self.irun - self.runs_with_small)
            opts['popsize'] = np.floor(self.popsize0 * self.popsize_multiplier ** (np.random.uniform() ** 2))
            opts['maxiter'] = min(self.maxiter0, 0.5 * sum(self.large_i) / opts['popsize'])
            # print('small basemul %s --> %s; maxiter %s' % (popsize_multiplier, opts['popsize'], opts['maxiter']))

        else:
            # A run with large population size; the population
            # doubling is implicit with incpopsize.
            self.poptype = 'large'

            self.popsize_multiplier = fmin_options['incpopsize'] ** (self.irun - self.runs_with_small)
            opts['popsize'] = self.popsize0 * self.popsize_multiplier
            opts['maxiter'] = self.maxiter0

        # recover from a CMA object
        if self.irun == 0 and isinstance(self.x0, CMAEvolutionStrategy):
            self.es = x0
            self.x0 = es.inputargs['x0']  # for the next restarts
            if np.isscalar(self.sigma0) and np.isfinite(self.sigma0) and self.sigma0 > 0:
                es.sigma = self.sigma0
            # debatable whether this makes sense:
            self.sigma0 = es.inputargs['sigma0']  # for the next restarts
            if self.options is not None:
                self.es.opts.set(options)
            # ignore further input args and keep original options

        else:  # default case
            if self.irun and eval(str(fmin_options['restart_from_best'])):
                utils.print_warning('CAVE: restart_from_best is often not useful',
                                    verbose=opts['verbose'])
                es = CMAEvolutionStrategy(self.best.x, self.sigma_factor * self.sigma0, opts)
            else:
                es = CMAEvolutionStrategy(self.x0.X, self.sigma_factor * self.sigma0, opts)

            # return opts, es
            if (self.eval_initial_x
                    or es.opts['CMA_elitist'] == 'initial'
                    or (es.opts['CMA_elitist'] and
                        self.eval_initial_x is None)):
                x = es.gp.pheno(es.mean,
                                into_bounds=es.boundary_handler.repair,
                                archive=es.sent_solutions)

                es.f0 = self.evaluate(X)

                es.best.update([x], es.sent_solutions, [es.f0], 1)
                es.countevals += 1

        opts = es.opts  # processed options, unambiguous
        # a hack:
        self.fmin_opts = CMAOptions("unchecked", **self.fmin_options.copy())
        for k in self.fmin_opts:
            # locals() cannot be modified directly, exec won't work
            # in 3.x, therefore
            self.fmin_opts.eval(k, loc={'N': es.N, 'popsize': opts['popsize']}, correct_key=False)

        es.logger.append = opts['verb_append'] or es.countiter > 0 or self.irun > 0
        # es.logger is "the same" logger, because the "identity"
        # is only determined by the `verb_filenameprefix` option
        self.logger = es.logger  # shortcut

        if self.noise_handler:
            if isinstance(noise_handler, type):
                self.noisehandler = noise_handler(es.N)
            else:
                self.noisehandler = noise_handler
            self.noise_handling = True
            if fmin_opts['noise_change_sigma_exponent'] > 0:
                es.opts['tolfacupx'] = inf
        else:
            self.noisehandler = ot.NoiseHandler(es.N, 0)  # switched off
            self.noise_handling = False

        self.es = es

    def infill(self):

        opts = self.opts
        es, gradf, parallelize = self.es, self.gradf, self.parallelize
        noise_handling, noisehandler = self.noise_handling, self.noisehandler

        # X, fit = eval_in_parallel(lambda: es.ask(1)[0], es.popsize, args, repetitions=noisehandler.evaluations-1)
        X, fit = es.ask_and_eval(gradf=gradf,
                                 evaluations=noisehandler.evaluations,
                                 aggregation=np.median,
                                 parallel_mode=parallelize)  # treats NaN with resampling if not parallel_mode

        if 11 < 3 and opts['vv']:  # inject a solution
            # use option check_point = [0]
            if 0 * np.random.randn() >= 0:
                X[0] = 0 + opts['vv'] * es.sigma ** 0 * np.random.randn(es.N)
                fit[0] = yield X[0]
                # print fit[0]

        if es.opts['verbose'] > 4:
            if es.countiter > 1 and min(fit) > es.best.last.f:
                self.unsuccessful_iterations_count += 1
                if self.unsuccessful_iterations_count > 4:
                    utils.print_message('%d unsuccessful iterations'
                                        % self.unsuccessful_iterations_count,
                                        iteration=es.countiter)
            else:
                self.unsuccessful_iterations_count = 0

        return Population.new(X=X)

    def _advance(self, off):
        opts, fmin_opts = self.opts, self.fmin_opts
        es, gradf, parallelize = self.es, self.gradf, self.parallelize
        noise_handling, noisehandler = self.noise_handling, self.noisehandler

        if es.stop():

            if opts['eval_final_mean']:
                mean_pheno = es.gp.pheno(es.mean,
                                         into_bounds=es.boundary_handler.repair,
                                         archive=es.sent_solutions)

                # evaluated
                fmean = yield mean_pheno

                es.countevals += 1
                es.best.update([mean_pheno], es.sent_solutions, [fmean], es.countevals)

            self.best.update(es.best, es.sent_solutions)  # in restarted case
            # es.best.update(best)

            this_evals = es.countevals - self.base_evals
            base_evals = es.countevals

            # BIPOP stats update

            if self.irun == 0:
                popsize0 = opts['popsize']
                maxiter0 = opts['maxiter']
                # XXX: This might be a bug? Reproduced from Matlab
                # small_i.append(this_evals)

            if self.bipop:
                if self.poptype == 'small':
                    self.small_i.append(this_evals)
                else:  # poptype == 'large'
                    self.large_i.append(this_evals)

            # final message
            if opts['verb_disp']:
                es.result_pretty(self.irun, time.asctime(time.localtime()), self.best.f)

            self.irun += 1
            # if irun > fmin_opts['restarts'] or 'ftarget' in es.stop() \
            # if irun > restarts or 'ftarget' in es.stop() \
            self.all_stoppings.append(dict(es.stop(check=False)))  # keeping the order

            # check if we are done running the algorithm
            if self.irun - self.runs_with_small > self.fmin_opts['restarts'] or 'ftarget' in es.stop() \
                    or 'maxfevals' in es.stop(check=False) or 'callback' in es.stop(check=False):
                self.termination.force_termination = True
                return

            opts['verb_append'] = es.countevals
            opts['popsize'] = fmin_opts['incpopsize'] * es.sp.popsize  # TODO: use rather options?
            try:
                opts['seed'] += 1
            except TypeError:
                pass

            # while irun

            # es.out['best'] = best  # TODO: this is a rather suboptimal type for inspection in the shell
            if self.irun:
                es.best.update(self.best)

        X = off.get("X")
        fit = self.evaluate(X)

        es.tell(X, fit)  # prepare for next iteration

        # if noise_handling:  # it would be better to also use these f-evaluations in tell
        #     es.sigma *= noisehandler(X, fit, objective_function, es.ask,
        #                              args=args) ** fmin_opts['noise_change_sigma_exponent']
        #
        #     es.countevals += noisehandler.evaluations_just_done  # TODO: this is a hack, not important though
        #     # es.more_to_write.append(noisehandler.evaluations_just_done)
        #     if noisehandler.maxevals > noisehandler.minevals:
        #         es.more_to_write.append(noisehandler.evaluations)
        #     if 1 < 3:
        #         # If sigma was above multiplied by the same
        #         #  factor cmean is divided by here, this is
        #         #  like only multiplying kappa instead of
        #         #  changing cmean and sigma.
        #         es.sp.cmean *= np.exp(-self.noise_kappa_exponent * np.tanh(noisehandler.noiseS))
        #         es.sp.cmean[es.sp.cmean > 1] = 1.0  # also works with "scalar arrays" like np.array(1.2)
        #

        for f in self.callback:
            f is None or f(es)
        es.disp()
        self.logger.add(
            # more_data=[noisehandler.evaluations, 10**noisehandler.noiseS] if noise_handling else [],
            modulo=1 if es.stop() and self.logger.modulo else None)
        if (opts['verb_log'] and opts['verb_plot'] and
                (es.countiter % max(opts['verb_plot'], opts['verb_log']) == 0 or es.stop())):
            self.logger.plot(324)

    def evaluate(self, x):
        pop = self.evaluator(self.problem, x)

        # set infeasible individual's objective values to np.nan - then CMAES can handle it
        for ind in pop:
            if not ind.feasible[0]:
                ind.F[0] = np.nan

        return pop.get("F")

    def _initialize(self):
        self._restart()
        return self.infill()

    def _set_optimum(self):
        val = self.pop
        if self.opt is not None:
            val = Population.merge(val, self.opt)
        self.opt = filter_optimum(val, least_infeasible=True)
    #
    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     del state["es"]
    #     return state
    #
    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     self.ers = None


class BIPOPCMAES(CMAES):

    def __init__(self, restarts=4, **kwargs):
        super().__init__(restarts=restarts, bipop=True, **kwargs)


parse_doc_string(CMAES.__init__)
