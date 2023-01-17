import numpy as np
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.survival import Survival, split_by_feasibility
from pymoo.core.population import Population
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function


class RankAndCrowding(Survival):

    def __init__(self, nds=None, crowding_func="cd"):
        """
        A generalization of the NSGA-II survival operator that ranks individuals by dominance criteria
        and sorts the last front by some user-specified crowding metric. The default is NSGA-II's crowding distances
        although others might be more effective.

        For many-objective problems, try using 'mnn' or '2nn'.

        For Bi-objective problems, 'pcd' is very effective.

        Parameters
        ----------
        nds : str or None, optional
            Pymoo type of non-dominated sorting. Defaults to None.

        crowding_func : str or callable, optional
            Crowding metric. Options are:

                - 'cd': crowding distances
                - 'pcd' or 'pruning-cd': improved pruning based on crowding distances
                - 'ce': crowding entropy
                - 'mnn': M-Neaest Neighbors
                - '2nn': 2-Neaest Neighbors

            If callable, it has the form ``fun(F, filter_out_duplicates=None, n_remove=None, **kwargs)``
            in which F (n, m) and must return metrics in a (n,) array.

            The options 'pcd', 'cd', and 'ce' are recommended for two-objective problems, whereas 'mnn' and '2nn' for many objective.
            When using 'pcd', 'mnn', or '2nn', individuals are already eliminated in a 'single' manner. 
            Due to Cython implementation, they are as fast as the corresponding 'cd', 'mnn-fast', or '2nn-fast', 
            although they can singnificantly improve diversity of solutions.
            Defaults to 'cd'.
        """

        crowding_func_ = get_crowding_function(crowding_func)

        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.crowding_func = crowding_func_

    def _do(self,
            problem,
            pop,
            *args,
            n_survive=None,
            **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):
            
            I = np.arange(len(front))

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(I) > n_survive:

                # Define how many will be removed
                n_remove = len(survivors) + len(front) - n_survive

                # re-calculate the crowding distance of the front
                crowding_of_front = \
                    self.crowding_func.do(
                        F[front, :],
                        n_remove=n_remove
                    )

                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:-n_remove]

            # otherwise take the whole front unsorted
            else:
                # calculate the crowding distance of the front
                crowding_of_front = \
                    self.crowding_func.do(
                        F[front, :],
                        n_remove=0
                    )

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]


class ConstrRankAndCrowding(Survival):

    def __init__(self, nds=None, crowding_func="cd"):
        """
        The Rank and Crowding survival approach for handling constraints proposed on
        GDE3 by Kukkonen, S. & Lampinen, J. (2005).

        Parameters
        ----------
        nds : str or None, optional
            Pymoo type of non-dominated sorting. Defaults to None.

        crowding_func : str or callable, optional
            Crowding metric. Options are:

                - 'cd': crowding distances
                - 'pcd' or 'pruning-cd': improved pruning based on crowding distances
                - 'ce': crowding entropy
                - 'mnn': M-Neaest Neighbors
                - '2nn': 2-Neaest Neighbors

            If callable, it has the form ``fun(F, filter_out_duplicates=None, n_remove=None, **kwargs)``
            in which F (n, m) and must return metrics in a (n,) array.

            The options 'pcd', 'cd', and 'ce' are recommended for two-objective problems, whereas 'mnn' and '2nn' for many objective.
            When using 'pcd', 'mnn', or '2nn', individuals are already eliminated in a 'single' manner. 
            Due to Cython implementation, they are as fast as the corresponding 'cd', 'mnn-fast', or '2nn-fast', 
            although they can singnificantly improve diversity of solutions.
            Defaults to 'cd'.
        """

        super().__init__(filter_infeasible=False)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.ranking = RankAndCrowding(nds=nds, crowding_func=crowding_func)

    def _do(self,
            problem,
            pop,
            *args,
            n_survive=None,
            **kwargs):

        if n_survive is None:
            n_survive = len(pop)

        n_survive = min(n_survive, len(pop))

        # If the split should be done beforehand
        if problem.n_constr > 0:

            # Split by feasibility
            feas, infeas = split_by_feasibility(pop, sort_infeas_by_cv=True, sort_feas_by_obj=False, return_pop=False)

            # Obtain len of feasible
            n_feas = len(feas)

            # Assure there is at least_one survivor
            if n_feas == 0:
                survivors = Population()
            else:
                survivors = self.ranking.do(problem, pop[feas], *args, n_survive=min(len(feas), n_survive), **kwargs)

            # Calculate how many individuals are still remaining to be filled up with infeasible ones
            n_remaining = n_survive - len(survivors)

            # If infeasible solutions need to be added
            if n_remaining > 0:

                # Constraints to new ranking
                G = pop[infeas].get("G")
                G = np.maximum(G, 0)
                H = pop[infeas].get("H")
                H = np.absolute(H)
                C = np.column_stack((G, H))

                # Fronts in infeasible population
                infeas_fronts = self.nds.do(C, n_stop_if_ranked=n_remaining)

                # Iterate over fronts
                for k, front in enumerate(infeas_fronts):

                    # Save ranks
                    pop[infeas][front].set("cv_rank", k)

                    # Current front sorted by CV
                    if len(survivors) + len(front) > n_survive:

                        # Obtain CV of front
                        CV = pop[infeas][front].get("CV").flatten()
                        I = randomized_argsort(CV, order='ascending', method='numpy')
                        I = I[:(n_survive - len(survivors))]

                    # Otherwise take the whole front unsorted
                    else:
                        I = np.arange(len(front))

                    # extend the survivors by all or selected individuals
                    survivors = Population.merge(survivors, pop[infeas][front[I]])

        else:
            survivors = self.ranking.do(problem, pop, *args, n_survive=n_survive, **kwargs)

        return survivors
