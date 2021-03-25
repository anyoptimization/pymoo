import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.docs import parse_doc_string
from pymoo.model.population import Population
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.optimum import filter_optimum
from pymoo.util.termination.default import SingleObjectiveDefaultTermination


class ES(GeneticAlgorithm):

    def __init__(self,
                 n_offsprings=200,
                 pop_size=None,
                 rule=1.0 / 7.0,
                 phi=1.0,
                 sampling=FloatRandomSampling(),
                 survival=FitnessSurvival(),
                 display=SingleObjectiveDisplay(),
                 **kwargs):

        if pop_size is None and n_offsprings is not None:
            pop_size = int(np.math.ceil(n_offsprings * rule))
        elif n_offsprings is None and pop_size is not None:
            n_offsprings = int(np.math.fllor(n_offsprings / rule))

        assert pop_size is not None and n_offsprings is not None, "You have to at least provivde pop_size of n_offsprings."
        assert n_offsprings >= 2 * pop_size, "The number of offsprings should be at least double the population size."

        super().__init__(pop_size=pop_size,
                         n_offsprings=n_offsprings,
                         sampling=sampling,
                         survival=survival,
                         display=display,
                         advance_after_initial_infill=True,
                         **kwargs)

        self.default_termination = SingleObjectiveDefaultTermination()
        self.phi = phi
        self.sigma_max = None
        self.tau, self.taup = None, None

    def _setup(self, problem, **kwargs):
        n = problem.n_var
        self.taup = self.phi / ((2 * n) ** 0.5)
        self.tau = self.phi / ((2 * (n ** 0.5)) ** 0.5)

        xl, xu = self.problem.bounds()
        self.sigma_max = (xu - xl) / (self.problem.n_var ** 0.5)

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills=infills, **kwargs)

        # initialize all individuals with the maximum sigma value
        infills.set("sigma", [self.sigma_max] * len(infills))

    def _infill(self):
        pop = self.pop
        n, (xl, xu) = self.problem.n_var, self.problem.bounds()

        # the values form the current population
        X, sigma = pop.get("X", "sigma")

        # cycle through the elites individuals for create the solutions
        I = np.arange(self.n_offsprings) % self.pop_size

        # transform X and sigma to the shape of number of offsprings
        X, sigma = X[I], sigma[I]

        # get the sigma only of the elites to be used
        sigma = es_intermediate_recomb(sigma)

        # calculate the new sigma based on tau and tau prime
        sigma = es_sigma(sigma, self.tau, self.taup)

        # make sure none of the sigmas exceeds the maximum
        sigma = np.minimum(sigma, self.sigma_max)

        # execute the evolutionary strategy to calculate the offspring solutions
        Xp = X + sigma * np.random.normal(size=sigma.shape)

        # repair the individuals which are not feasible by sampling from sigma again
        Xp = es_mut_repair(Xp, X, sigma, xl, xu, 10)

        # create the population to proceed further
        off = Population.new(X=Xp, sigma=sigma)

        return off

    def _advance(self, infills=None, **kwargs):
        self.pop = self.survival.do(self.problem, infills, n_survive=self.pop_size)

    def _set_optimum(self):
        pop = self.pop if self.opt is None else Population.merge(self.opt, self.pop)
        self.opt = filter_optimum(pop, least_infeasible=True)


def es_sigma(sigma, tau, taup):
    _lambda, _n = sigma.shape
    return sigma * np.exp(taup * np.random.normal(size=(_lambda, 1)) + tau * np.random.normal(size=(_lambda, _n)))


def es_intermediate_recomb(sigma):
    _lambda, _n = sigma.shape
    sigma_hat = np.zeros_like(sigma)

    for i in range(_lambda):
        for j in range(_n):
            k = np.random.randint(_lambda)
            sigma_hat[i, j] = (sigma[i, j] + sigma[k, j]) / 2.0

    return sigma_hat


def es_mut_repair(Xp, X, sigma, xl, xu, n_trials):

    # reshape xl and xu to be the same shape as the input
    XL = xl[None, :].repeat(len(Xp), axis=0)
    XU = xu[None, :].repeat(len(Xp), axis=0)

    all_in_bounds = False

    # for the given number of trials
    for k in range(n_trials):

        # find all indices which are out of bounds
        i, j = np.where(np.logical_or(Xp < XL, Xp > XU))

        if len(i) == 0:
            all_in_bounds = True
            break
        else:
            # do the mutation again vectored for all values not in bound
            Xp[i, j] = X[i, j] + sigma[i, j] * np.random.normal(size=len(i))

    # if there are still solutions which boundaries are violated, set them to the original X
    if not all_in_bounds:
        i, j = np.where(np.logical_or(Xp < XL, Xp > XU))
        Xp[i, j] = X[i, j]

    return Xp


def es_mut_loop(X, sigmap, xl, xu, n_trials=10):
    _lambda, _n = sigmap.shape

    # X prime which will be returned by the algorithm (set the default value to the same as parent)
    Xp = np.zeros_like(sigmap)

    # for each of the new offsprings
    for i in range(_lambda):

        # for each variable of it
        for j in range(_n):

            # by default just copy the value if no value is in bounds this will stay
            Xp[i, j] = X[i, j]

            # try to set the value a few time and be done if in bounds
            for _ in range(n_trials):

                # calculate the mutated value
                x = X[i, j] + sigmap[i, j] * np.random.normal()

                # if it is inside the bounds accept it - otherwise try again
                if xl[j] <= x <= xu[j]:
                    Xp[i, j] = x
                    break

    return Xp


parse_doc_string(ES.__init__)
