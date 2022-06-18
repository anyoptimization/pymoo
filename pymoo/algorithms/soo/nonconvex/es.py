import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.population import Population
from pymoo.docs import parse_doc_string
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.util.optimum import filter_optimum


class ES(GeneticAlgorithm):

    def __init__(self,
                 n_offsprings=200,
                 pop_size=None,
                 rule=1.0 / 7.0,
                 phi=1.0,
                 gamma=0.85,
                 sampling=FloatRandomSampling(),
                 survival=FitnessSurvival(),
                 output=SingleObjectiveOutput(),
                 **kwargs):

        """
        Evolutionary Strategy (ES)

        Parameters
        ----------
        n_offsprings : int
            The number of individuals created in each iteration.
        pop_size : int
            The number of individuals which are surviving from the offspring population (non-elitist)
        rule : float
            The rule (ratio) of individuals surviving. This automatically either calculated `n_offsprings` or `pop_size`.
        phi : float
            Expected rate of convergence (usually 1.0).
        gamma : float
            If not `None`, some individuals are created using the differentials with this as a length scale.
        sampling : object
            The sampling method for creating the initial population.
        """

        if pop_size is None and n_offsprings is not None:
            pop_size = int(np.math.ceil(n_offsprings * rule))
        elif n_offsprings is None and pop_size is not None:
            n_offsprings = int(np.math.floor(n_offsprings / rule))

        assert pop_size is not None and n_offsprings is not None, "You have to at least provivde pop_size of n_offsprings."
        assert n_offsprings >= 2 * pop_size, "The number of offsprings should be at least double the population size."

        super().__init__(pop_size=pop_size,
                         n_offsprings=n_offsprings,
                         sampling=sampling,
                         survival=survival,
                         output=output,
                         advance_after_initial_infill=True,
                         **kwargs)

        self.termination = DefaultSingleObjectiveTermination()
        self.phi = phi
        self.gamma = gamma

        self.tau, self.taup, self.sigma_max = None, None, None

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
        pop, mu, _lambda = self.pop, self.pop_size, self.n_offsprings
        xl, xu = self.problem.bounds()
        X, sigma = pop.get("X", "sigma")

        # cycle through the elites individuals for create the solutions
        I = np.arange(_lambda) % mu

        # transform X and sigma to the shape of number of offsprings
        X, sigma = X[I], sigma[I]

        # get the sigma only of the elites to be used
        sigmap = es_intermediate_recomb(sigma)

        # calculate the new sigma based on tau and tau prime
        sigmap = np.minimum(self.sigma_max, es_sigma(sigmap, self.tau, self.taup))

        # execute the evolutionary strategy to calculate the offspring solutions
        Xp = X + sigmap * np.random.normal(size=sigmap.shape)

        # if gamma is not none do the differential variation overwrite Xp and sigmap for the first mu-1 individuals
        if self.gamma is not None:
            Xp[:mu - 1] = X[:mu - 1] + self.gamma * (X[0] - X[1:mu])
            sigmap[:mu - 1] = sigma[:mu - 1]

        # if we have bounds to consider -> repair the individuals which are out of bounds
        if self.problem.has_bounds():
            Xp = es_mut_repair(Xp, X, sigmap, xl, xu, 10)

        # create the population to proceed further
        off = Population.new(X=Xp, sigma=sigmap)

        return off

    def _advance(self, infills=None, **kwargs):

        # if not all solutions suggested by infill() are evaluated we create a more semi (mu+lambda) algorithm
        if len(infills) < self.pop_size:
            infills = Population.merge(infills, self.pop)

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
