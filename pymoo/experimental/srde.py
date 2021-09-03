import numpy as np

from pymoo.algorithms.soo.nonconvex.de import DE, DifferentialEvolutionMating
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


# =========================================================================================================
# Implementation
# =========================================================================================================


class SRDE(DE):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _infill(self):
        pop = self.pop

        f = pop.get("F")[:, 0]
        cv = pop.get("CV")[:, 0]

        # print((cv == 0).sum())

        S = np.arange(len(pop))

        feas = (cv == 0)
        n_feas = feas.sum()

        if feas.sum() > 0:
            f_min, cv_min = f[feas].min(), 0.0
        else:
            f_min, cv_min = np.inf, cv[~feas].min()

        if n_feas > 0:

            above_or_equal_fmin = np.where(f >= f_min)[0]
            below_fmin = np.where(f < f_min)[0]

            balance = len(pop) // 2

            # less feasible solutions than desired
            if n_feas < balance:

                n_repl = balance - n_feas

                if len(above_or_equal_fmin) > 0:
                    sort_by_dec_cv = below_fmin[cv[below_fmin].argsort()[::-1]]
                    repl = sort_by_dec_cv[:n_repl]
                    S[repl] = np.random.choice(above_or_equal_fmin, size=len(repl))

            else:

                if len(below_fmin) > 0:

                    n_repl = n_feas - balance

                    sort_by_dec_cv = above_or_equal_fmin[f[above_or_equal_fmin].argsort()[::-1]]
                    repl = sort_by_dec_cv[:n_repl]

                    S[repl] = np.random.choice(below_fmin, size=len(repl))


        mating = DifferentialEvolutionMating()
        infills = mating.do(self.problem, self.pop, self.n_offsprings, S=S, algorithm=self)

        return infills

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, "This algorithms uses the AskAndTell interface thus infills must to be provided."

        pop, off = self.pop, infills
        assert len(pop) == len(off)

        pop_X, pop_f, pop_cv = pop.get("X", "F", "CV")
        pop_f, pop_cv = pop_f[:, 0], pop_cv[:, 0]

        off_X, off_f, off_cv = off.get("X", "F", "CV")
        off_f, off_cv = off_f[:, 0], off_cv[:, 0]

        pop_feas, off_feas = pop_cv == 0, off_cv == 0

        f, cv, feas = np.row_stack([pop_f, off_f]), np.row_stack([pop_cv, off_cv]), np.row_stack([pop_feas, off_feas])

        if feas.sum() > 0:
            f_min, cv_min = f[feas].min(), 0.0
        else:
            f_min, cv_min = np.inf, cv[~feas].min()

        # import matplotlib.pyplot as plt
        #
        # plt.scatter(pop_cv[pop_feas], pop_f[pop_feas], facecolor="none", edgecolors="blue")
        # plt.scatter(pop_cv[~pop_feas], pop_f[~pop_feas], facecolor="none", edgecolors="red")
        #
        # plt.scatter(off_cv[off_feas], off_f[off_feas], facecolor="none", edgecolors="purple", alpha=0.5)
        # plt.scatter(off_cv[~off_feas], off_f[~off_feas], facecolor="none", edgecolors="orange", alpha=0.5)

        for k in range(len(pop)):

            if pop_f[k] >= f_min:

                # if off_f[k] <= pop_f[k]:
                #     self.pop[k] = infills[k]

                if off_cv[k] < pop_cv[k]:
                    self.pop[k] = infills[k]
                elif off_cv[k] == pop_cv[k] and off_f[k] < pop_f[k]:
                    self.pop[k] = infills[k]

            else:

                if off_cv[k] <= pop_cv[k]:
                    self.pop[k] = infills[k]

        # F, CV = self.pop.get("F", "CV")
        # f, cv = F[:, 0], CV[:, 0]
        # plt.scatter(cv, f, facecolor="none", edgecolors="green", s=3, alpha=0.9)
        # plt.show()

    def _advance6(self, infills=None, **kwargs):
        assert infills is not None, "This algorithms uses the AskAndTell interface thus infills must to be provided."

        pop, off = self.pop, infills
        assert len(pop) == len(off)

        N = len(pop)
        pop_X, pop_f, pop_cv = pop.get("X", "F", "CV")
        pop_f, pop_cv = pop_f[:, 0], pop_cv[:, 0]

        off_X, off_f, off_cv = off.get("X", "F", "CV")
        off_f, off_cv = off_f[:, 0], off_cv[:, 0]

        pop_feas, off_feas = pop_cv == 0, off_cv == 0

        f, cv = np.concatenate([pop_f, off_f]), np.concatenate([pop_cv, off_cv])
        nds = NonDominatedSorting().do(np.column_stack([cv, f]), only_non_dominated_front=True)

        pop_cv_nds = set([k for k in nds if k < len(pop)]),
        off_cv_nds = set([k % len(pop) for k in nds if k >= len(pop)])

        import matplotlib.pyplot as plt

        plt.scatter(cv[nds], f[nds], marker="x", color="black", s=30)

        plt.scatter(pop_cv[pop_feas], pop_f[pop_feas], facecolor="none", edgecolors="blue")
        plt.scatter(pop_cv[~pop_feas], pop_f[~pop_feas], facecolor="none", edgecolors="red")

        plt.scatter(off_cv[off_feas], off_f[off_feas], facecolor="none", edgecolors="purple", alpha=0.5)
        plt.scatter(off_cv[~off_feas], off_f[~off_feas], facecolor="none", edgecolors="orange", alpha=0.5)

        hierarch_sort = hierarchical_sort(pop_f, pop_cv)
        fittest = hierarch_sort[0]
        topx = set(hierarch_sort[:10])

        infeas = np.sort(pop_cv[~pop_feas])
        n_infeas = len(pop) // 2

        if len(infeas) == 0:
            eps = np.inf
        else:
            k = min(n_infeas, len(infeas) - 1)
            eps = infeas[k]

        for k in range(len(pop)):

            if k in off_cv_nds:
                if k in pop_cv_nds:
                    if off_cv[k] <= pop_cv[k]:
                        self.pop[k] = infills[k]
                else:
                    if k != fittest:
                        self.pop[k] = infills[k]

            elif k in topx:

                if pop_cv[k] > 0 and off_cv[k] <= pop_cv[k]:
                    self.pop[k] = infills[k]

                else:
                    if off_cv[k] == 0 and off_f[k] <= pop_f[k]:
                        self.pop[k] = infills[k]

            else:

                if off_cv[k] <= eps and pop_cv[k] <= eps:
                    if off_f[k] <= pop_f[k]:
                        self.pop[k] = infills[k]

                elif off_cv[k] <= pop_cv[k]:
                    self.pop[k] = infills[k]

        plt.plot(np.ones(1000) * eps, np.linspace(f.min(), f.max(), 1000), "--", color="black")

        F, CV = self.pop.get("F", "CV")
        f, cv = F[:, 0], CV[:, 0]
        plt.scatter(cv, f, facecolor="none", edgecolors="green", s=3, alpha=0.9)

        plt.xlim(0, cv.max())
        plt.show()

    def _advance3(self, infills=None, **kwargs):
        assert infills is not None, "This algorithms uses the AskAndTell interface thus infills must to be provided."

        pop, off = self.pop, infills
        assert len(pop) == len(off)

        N = len(pop)
        pop_X, pop_f, pop_cv = pop.get("X", "F", "CV")
        pop_f, pop_cv = pop_f[:, 0], pop_cv[:, 0]

        off_X, off_f, off_cv = off.get("X", "F", "CV")
        off_f, off_cv = off_f[:, 0], off_cv[:, 0]

        pop_feas, off_feas = pop_cv == 0, off_cv == 0

        infeas = np.sort(pop_cv[~pop_feas])
        n_infeas = len(pop) // 2
        if len(infeas) == 0:
            eps = np.inf
        elif len(infeas) < n_infeas:
            eps = infeas[n_infeas]

        if pop_feas.sum() > 0:
            f_min, cv_min = pop_f[pop_feas].min(), 0.0
        else:
            f_min, cv_min = np.inf, pop_cv[~pop_feas].min()

        f, cv = np.concatenate([pop_f, off_f]), np.concatenate([pop_cv, off_cv])
        nds = NonDominatedSorting().do(np.column_stack([cv, f]), only_non_dominated_front=True)

        # if pop_feas.sum() > 0:
        #     nds = [k for k in nds if cv[k] < 0.001]

        pop_cv_nds = set([k for k in nds if k < len(pop)]),
        off_cv_nds = set([k % len(pop) for k in nds if k >= len(pop)])

        n_infeas = (~pop_feas).sum()
        n_infeas_goal = int(self.r_infeas * N)

        repl = np.full(N, False)

        for k in range(N):

            if pop_feas[k] and off_feas[k]:
                if off_f[k] <= pop_f[k]:
                    repl[k] = True

            elif pop_feas[k] and ~off_feas[k]:

                if pop_f[k] > f_min:
                    if k in off_cv_nds:
                        repl[k] = True

            elif ~pop_feas[k] and off_feas[k]:

                if k not in pop_cv_nds:
                    repl[k] = True

            elif ~pop_feas[k] and ~off_feas[k]:

                if k in off_cv_nds:

                    if k in pop_cv_nds:

                        if off_cv[k] <= pop_cv[k]:
                            repl[k] = True

                    else:
                        repl[k] = True

                else:
                    if off_cv[k] <= pop_cv[k]:
                        repl[k] = True

        print(repl.sum())

        import matplotlib.pyplot as plt

        plt.scatter(cv[nds], f[nds], marker="x", color="black", s=30)

        F, CV = self.pop.get("F", "CV")
        f, cv = F[:, 0], CV[:, 0]
        feas = cv <= 0

        plt.scatter(cv[feas], f[feas], facecolor="none", edgecolors="blue")
        plt.scatter(cv[~feas], f[~feas], facecolor="none", edgecolors="red")

        F, CV = infills.get("F", "CV")
        f, cv = F[:, 0], CV[:, 0]
        feas = cv <= 0

        plt.scatter(cv[feas], f[feas], facecolor="none", edgecolors="blue", alpha=0.2)
        plt.scatter(cv[~feas], f[~feas], facecolor="none", edgecolors="red", alpha=0.2)

        self.pop[repl] = off[repl]

        F, CV = self.pop.get("F", "CV")
        f, cv = F[:, 0], CV[:, 0]
        plt.scatter(cv, f, facecolor="none", edgecolors="green", s=3, alpha=0.9)

        plt.show()

    def _advance2(self, infills=None, **kwargs):
        assert infills is not None, "This algorithms uses the AskAndTell interface thus infills must to be provided."
        pop, off = self.pop, infills
        N = len(pop)

        if self.t_infeas is None:
            self.t_infeas = np.zeros(N)

        assert len(pop) == len(off)

        pop_f, pop_cv, pop_t = pop.get("F")[:, 0], pop.get("CV")[:, 0], self.n_gen - pop.get("n_gen") - 1
        off_f, off_cv = off.get("F")[:, 0], off.get("CV")[:, 0]
        pop_feas, off_feas = pop_cv == 0, off_cv == 0

        self.t_infeas += 1
        self.t_infeas[pop_feas] = 0

        if pop_feas.sum() > 0:
            f_min, cv_min = pop_f[pop_feas].min(), 0.0
        else:
            f_min, cv_min = np.inf, pop_cv[~pop_feas].min()

        n_infeas = (pop_cv > 0).sum()
        n_goal_infeas = int(N * self.r_infeas)
        n_goal_feas = N - n_goal_infeas

        # if all solutions are currently infeasible
        if pop_feas.sum() == 0:
            for k in range(N):
                if off_cv[k] <= pop_cv[k]:
                    repl[k] = True

        else:

            # S = off_f.argsort()

            for k in range(N):

                if pop_feas[k]:

                    # if both are feasible replace if function value is better
                    if off_feas[k]:
                        if off_f[k] <= pop_f[k]:
                            repl[k] = True

                    # current solution is feasible and offspring is not
                    else:

                        # never replace the current best solution
                        if pop_f[k] > f_min:

                            # only consider a replacement to satisfy the ration of infeasible solutions in pop
                            if n_infeas < n_goal_infeas:
                                if off_f[k] <= pop_f[k]:
                                    repl[k] = True
                                    n_infeas += 1

                # the current solution is infeasible
                else:

                    # we have found a new solution which is feasible and improves the current best
                    if off_feas[k] and off_f[k] <= f_min:
                        repl[k] = True

                    # otherwise we apply our fuzzy logic
                    else:

                        n_feas = N - n_infeas

                        if n_feas < n_goal_feas and off_cv[k] < pop_cv[k]:
                            repl[k] = True
                            n_infeas -= 1

                        # we aim to find a solution which improve f and cv at the same time
                        elif self.t_infeas[k] <= 10:

                            if off_cv[k] < pop_cv[k] and off_f[k] < pop_f[k]:
                                repl[k] = True

                        else:

                            if off_cv[k] < pop_cv[k]:
                                repl[k] = True

        # self.pop[self.indices] = self.survival.do(self.problem, self.pop[self.indices], infills, algorithm=self)

        import matplotlib.pyplot as plt

        F, CV = self.pop.get("F", "CV")
        f, cv = F[:, 0], CV[:, 0]
        feas = cv <= 0

        plt.scatter(cv[feas], f[feas], facecolor="none", edgecolors="blue")
        plt.scatter(cv[~feas], f[~feas], facecolor="none", edgecolors="red")

        F, CV = infills.get("F", "CV")
        f, cv = F[:, 0], CV[:, 0]
        feas = cv <= 0

        plt.scatter(cv[feas], f[feas], facecolor="none", edgecolors="blue", alpha=0.2)
        plt.scatter(cv[~feas], f[~feas], facecolor="none", edgecolors="red", alpha=0.2)

        plt.show()

        self.pop = pop

        # f = self.pop.get("F")[:, 0]
        # cv = self.pop.get("CV")[:, 0]
        #
        # self.pop = self.pop[stochastic_ranking(f, cv, 0.45)]

        # sort the population by fitness to make the selection simpler for mating (not an actual survival, just sorting)
        # self.pop = FitnessSurvival().do(self.problem, self.pop)


if __name__ == "__main__":

    problem = get_problem("g01")

    algorithm = SRDE(pop_size=20, CR=0.3)

    # algorithm = DE(pop_size=100, variant="DE/rand/1/bin", CR=0.5)

    res = minimize(problem,
                   algorithm,
                   ("n_gen", 3000),
                   seed=1,
                   verbose=True)

    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
