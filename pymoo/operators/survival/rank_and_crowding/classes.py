"""Rank and crowding based survival operators."""

import numpy as np

from pymoo.core.population import Population
from pymoo.core.survival import Survival, split_by_feasibility
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort


class RankAndCrowding(Survival):
    """Rank and crowding based survival operator generalization of NSGA-II.

    Ranks individuals by dominance criteria and sorts the last front by a
    user-specified crowding metric. The default is NSGA-II's crowding distances.
    For many-objective problems, try 'mnn' or '2nn'. For bi-objective, 'pcd' is effective.
    """

    def __init__(self, nds=None, crowding_func: str = "cd") -> None:
        """Initialize the rank and crowding survival operator.

        Args:
            nds: Non-dominated sorting implementation. Defaults to None.
            crowding_func: Crowding metric function. Options: 'cd', 'pcd', 'pruning-cd',
                'ce', 'mnn', '2nn'. Can also be a callable with signature
                fun(F, filter_out_duplicates=None, n_remove=None, **kwargs).
                Defaults to 'cd'.

        Note:
            'pcd', 'cd', and 'ce' are recommended for two-objective problems.
            'mnn' and '2nn' are recommended for many-objective problems.
        """
        crowding_func_ = get_crowding_function(crowding_func)

        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.crowding_func = crowding_func_

    def _do(  # type: ignore[override]
        self,
        problem,
        pop,
        *args,
        random_state=None,
        n_survive=None,
        **kwargs,
    ) -> Population:
        """Select survivors using rank and crowding.

        Args:
            problem: Optimization problem.
            pop: Population.
            *args: Additional positional arguments.
            random_state: Random state for reproducibility.
            n_survive: Number of individuals to survive.
            **kwargs: Additional keyword arguments.

        Returns:
            Population of survivors.
        """
        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors: list[int] = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):
            I = np.arange(len(front))  # noqa: E741

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(I) > n_survive:
                # Define how many will be removed
                n_remove = len(survivors) + len(front) - n_survive

                # re-calculate the crowding distance of the front
                crowding_of_front = self.crowding_func.do(F[front, :], n_remove=n_remove)

                I = randomized_argsort(  # noqa: E741
                    crowding_of_front,
                    order="descending",
                    method="numpy",
                    random_state=random_state,
                )
                I = I[:-n_remove]  # noqa: E741

            # otherwise take the whole front unsorted
            else:
                # calculate the crowding distance of the front
                crowding_of_front = self.crowding_func.do(F[front, :], n_remove=0)

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]


class ConstrRankAndCrowding(Survival):
    """Rank and crowding based survival with constraint handling (GDE3).

    The Rank and Crowding survival approach for handling constraints proposed in
    GDE3 by Kukkonen, S. & Lampinen, J. (2005).
    """

    def __init__(self, nds=None, crowding_func: str = "cd") -> None:
        """Initialize the constrained rank and crowding survival operator.

        Args:
            nds: Non-dominated sorting implementation. Defaults to None.
            crowding_func: Crowding metric function. Options: 'cd', 'pcd', 'pruning-cd',
                'ce', 'mnn', '2nn'. Can also be a callable with signature
                fun(F, filter_out_duplicates=None, n_remove=None, **kwargs).
                Defaults to 'cd'.

        Note:
            'pcd', 'cd', and 'ce' are recommended for two-objective problems.
            'mnn' and '2nn' are recommended for many-objective problems.
        """
        super().__init__(filter_infeasible=False)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.ranking = RankAndCrowding(nds=nds, crowding_func=crowding_func)

    def _do(  # type: ignore[override]
        self, problem, pop, *args, n_survive=None, **kwargs
    ) -> Population:
        """Select survivors handling constraints and infeasibility.

        Args:
            problem: Optimization problem.
            pop: Population.
            *args: Additional positional arguments.
            n_survive: Number of individuals to survive.
            **kwargs: Additional keyword arguments.

        Returns:
            Population of survivors.
        """
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
                survivors = self.ranking.do(
                    problem,
                    pop[feas],
                    *args,
                    n_survive=min(len(feas), n_survive),
                    **kwargs,
                )

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
                        I = randomized_argsort(CV, order="ascending", method="numpy")  # noqa: E741
                        I = I[: (n_survive - len(survivors))]  # noqa: E741

                    # Otherwise take the whole front unsorted
                    else:
                        I = np.arange(len(front))  # noqa: E741

                    # extend the survivors by all or selected individuals
                    survivors = Population.merge(survivors, pop[infeas][front[I]])

        else:
            survivors = self.ranking.do(problem, pop, *args, n_survive=n_survive, **kwargs)

        return survivors
