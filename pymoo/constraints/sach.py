import numpy as np

from pymoo.constraints.tcv import TotalConstraintViolation
from pymoo.core.survival import Survival


class SelfAdaptiveConstraintSurvival(Survival):

    def __init__(self, adapt_norm=True) -> None:
        super().__init__(filter_infeasible=False)
        self.adapt_norm = adapt_norm
        self.ieq_scale, self.eq_scale = None, None

    def _do(self, problem, pop, n_survive=None, algorithm=None, **kwargs):

        F, G, H, feas = pop.get("f", "G", "H", "feasible")

        # the final penalized values to be used
        F_penalized = np.copy(F)

        # we only need to apply any penalty if there is at least one infeasible solution
        if not np.all(feas):

            # get the scales from the current population
            if self.adapt_norm:
                ieq_scale, eq_scale = G.max(axis=0), H.max(axis=0)
            else:
                if self.ieq_scale is None or self.eq_scale is None:
                    self.ieq_scale, self.eq_scale = G.max(axis=0), H.max(axis=0)
                ieq_scale, eq_scale = self.ieq_scale, self.eq_scale

            # print(ieq_scale, eq_scale)

            # calculate the weighted constrained violations
            CV = TotalConstraintViolation(ieq_scale=ieq_scale, eq_scale=eq_scale, aggr_func=np.mean).calc(G, H)
            # CV = TotalConstraintViolation(aggr_func=np.mean).calc(G, H)

            # split the population into two sub populations - feasible and infeasible - and sort them by F oder CV
            feas = np.where(CV <= 0.0)[0]
            feas = feas[np.argsort(F[feas])]

            infeas = np.where(CV > 0.0)[0]
            infeas = infeas[np.argsort(CV[infeas])]

            # find the best solution - if feasible solutions exist best obj, otherwise least infeasible
            best = feas[0] if len(feas) > 0 else infeas[0]

            # the maximum objective value no matter if feasible or infeasible
            highest = F.argmax()

            # infeasible solutions which are better than the best feasible
            infeas_obj_lower_than_best = infeas[F[infeas] < F[best]]

            # set the worst individual depending on the cases
            if len(infeas_obj_lower_than_best) > 0:
                # if there is at least one infeasible solution with a better function value than best -> take max CV
                worst = infeas_obj_lower_than_best[-1]
            else:
                # otherwise simply take the one with max CV
                worst = infeas[-1]

            # FIRST PENALTY
            if len(infeas_obj_lower_than_best) > 0:
                # calculate a normalized constraint violation for linear penalty
                ncv = (CV[infeas] - CV[best]) / (CV[worst] - CV[best])

                # now apply the penalty to all infeasible solutions that best has an equal or lower value than worst
                F_penalized[infeas] += ncv * (F[best] - F[worst])

            # import matplotlib.pyplot as plt
            #
            # plt.scatter(CV, F, facecolor='none', edgecolors='black', alpha=0.5)
            # plt.scatter(CV[best], F[best], label="best", marker="x", s=200, color="red", alpha=0.5)
            # plt.scatter(CV[worst], F[worst], label="worst", marker="p", s=200, color="green", alpha=0.5)
            # plt.scatter(CV[highest], F[highest], label="highest", marker="s", s=200, color="orange", alpha=0.5)
            #
            # plt.scatter(CV, F_penalized, facecolor='none', edgecolors='purple', alpha=0.5)
            # for k in range(len(pop)):
            #     plt.plot([CV[k], CV[k]], [F[k], F_penalized[k]], color="black", alpha=0.3, linewidth=0.5)

            # SECOND PENALTY
            if F[worst] == F[highest]:
                gamma = 0.0
            elif F[worst] <= F[best]:
                gamma = (F[highest] - F[best]) / np.abs(F[best])
            else:
                gamma = (F[highest] - F[worst]) / np.abs(F[worst])

            ncv = (np.exp(2.0 * CV[infeas]) - 1.0) / (np.exp(2.0) - 1.0)
            F_penalized[infeas] += gamma * np.abs(F_penalized[infeas]) * ncv

            # plt.scatter(CV, F_penalized, facecolor='none', edgecolors='red', alpha=0.5)
            # for k in range(len(pop)):
            #     plt.plot([CV[k], CV[k]], [F[k], F_penalized[k]], color="black", alpha=0.3, linewidth=0.5)
            #
            # cv_min, cv_max = -0.1, CV.max() * 1.1
            # plt.xlim(cv_min, cv_max)
            #
            # if CV[best] <= 0:
            #     plt.plot([cv_min, cv_max], [F[best], F[best]], "--", color="black", linewidth=0.7, alpha=0.8)

            default = np.lexsort([F, CV])
            F_penalized[default[:5]] = - np.inf

            # F_penalized[best] = - np.inf

            S = np.argsort(F_penalized)[:n_survive]

            # print(pop[best].CV, pop[best].F)

        else:
            S = np.argsort(F)[:n_survive]

        # plt.scatter(CV[S], F[S], marker="o", color="black", s=1)
        # plt.legend()
        # plt.show()
        # plt.close()

        return pop[S]


