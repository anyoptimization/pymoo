import matplotlib
matplotlib.use('Agg')

from pymoo.model.problem import Problem

import numpy as np


RESULT = None


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=2,
                         xl=np.array([-2, -2]),
                         xu=np.array([2, 2]))

    def _evaluate(self, X, out, *args, **kwargs):
        f1 = X[:, 0] ** 2 + X[:, 1] ** 2
        f2 = (X[:, 0] - 1) ** 2 + X[:, 1] ** 2

        g1 = 2 * (X[:, 0] - 0.1) * (X[:, 0] - 0.9) / 0.18
        g2 = - 20 * (X[:, 0] - 0.4) * (X[:, 0] - 0.6) / 4.8

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])


vectorized_problem = MyProblem()

import numpy as np
from pymoo.model.problem import Problem


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=2,
                         xl=np.array([-2, -2]),
                         xu=np.array([2, 2]),
                         elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[0] ** 2 + x[1] ** 2
        f2 = (x[0] - 1) ** 2 + x[1] ** 2

        g1 = 2 * (x[0] - 0.1) * (x[0] - 0.9) / 0.18
        g2 = - 20 * (x[0] - 0.4) * (x[0] - 0.6) / 4.8

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]


elementwise_problem = MyProblem()


from pymoo.model.problem import FunctionalProblem

objs = [
    lambda x: x[0]**2 + x[1]**2,
    lambda x: (x[0]-1)**2 + x[1]**2
]

constr_ieq = [
    lambda x: 2*(x[0]-0.1) * (x[0]-0.9) / 0.18,
    lambda x: - 20*(x[0]-0.4) * (x[0]-0.6) / 4.8
]

functional_problem = FunctionalProblem(2,
                                       objs,
                                       constr_ieq=constr_ieq,
                                       xl=np.array([-2,-2]),
                                       xu=np.array([2,2]))


def func_pf(flatten=True, **kwargs):
    f1_a = np.linspace(0.1 ** 2, 0.4 ** 2, 100)
    f2_a = (np.sqrt(f1_a) - 1) ** 2

    f1_b = np.linspace(0.6 ** 2, 0.9 ** 2, 100)
    f2_b = (np.sqrt(f1_b) - 1) ** 2

    a, b = np.column_stack([f1_a, f2_a]), np.column_stack([f1_b, f2_b])
    return stack(a, b, flatten=flatten)


def func_ps(flatten=True, **kwargs):
    x1_a = np.linspace(0.1, 0.4, 50)
    x1_b = np.linspace(0.6, 0.9, 50)
    x2 = np.zeros(50)

    a, b = np.column_stack([x1_a, x2]), np.column_stack([x1_b, x2])
    return stack(a, b, flatten=flatten)


import numpy as np
from pymoo.util.misc import stack


class MyTestProblem(MyProblem):

    def _calc_pareto_front(self, *args, **kwargs):
        return func_pf(**kwargs)

    def _calc_pareto_set(self, *args, **kwargs):
        return func_ps(**kwargs)


test_problem = MyTestProblem()


from pymoo.model.problem import FunctionalProblem


functional_test_problem = FunctionalProblem(2,
                                            objs,
                                            constr_ieq=constr_ieq,
                                            xl=-2,
                                            xu=2,
                                            func_pf=func_pf,
                                            func_ps=func_ps
                                            )


for problem in [vectorized_problem, elementwise_problem, functional_problem, test_problem, functional_test_problem]:
    from pymoo.algorithms.nsga2 import NSGA2
    from pymoo.factory import get_sampling, get_crossover, get_mutation


    print(problem.__class__.__name__)

    algorithm = NSGA2(
        pop_size=40,
        n_offsprings=10,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True
    )

    from pymoo.factory import get_termination

    termination = get_termination("n_gen", 40)

    from pymoo.optimize import minimize

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)

    if RESULT is None:
        RESULT = res
    else:
        np.testing.assert_allclose(RESULT.F, res.F)
        print("YES")

    from pymoo.visualization.scatter import Scatter

    # get the pareto-set and pareto-front for plotting
    ps = problem.pareto_set(use_cache=False, flatten=False)
    pf = problem.pareto_front(use_cache=False, flatten=False)

    # Design Space
    plot = Scatter(title="Design Space", axis_labels="x")
    plot.add(res.X, s=30, facecolors='none', edgecolors='r')
    if ps is not None:
        plot.add(ps, plot_type="line", color="black", alpha=0.7)
    plot.do()
    plot.apply(lambda ax: ax.set_xlim(-0.5, 1.5))
    plot.apply(lambda ax: ax.set_ylim(-2, 2))
    plot.show()

    # Objective Space
    plot = Scatter(title="Objective Space")
    plot.add(res.F)
    if pf is not None:
        plot.add(pf, plot_type="line", color="black", alpha=0.7)
    plot.show()

    n_evals = []  # corresponding number of function evaluations\
    F = []  # the objective space values in each generation
    cv = []  # constraint violation in each generation

    # iterate over the deepcopies of algorithms
    for algorithm in res.history:
        # store the number of function evaluations
        n_evals.append(algorithm.evaluator.n_eval)

        # retrieve the optimum from the algorithm
        opt = algorithm.opt

        # store the least contraint violation in this generation
        cv.append(opt.get("CV").min())

        # filter out only the feasible and append
        feas = np.where(opt.get("feasible"))[0]
        _F = opt.get("F")[feas]
        F.append(_F)

    import matplotlib.pyplot as plt

    k = min([i for i in range(len(cv)) if cv[i] <= 0])
    first_feas_evals = n_evals[k]
    print(f"First feasible solution found after {first_feas_evals} evaluations")

    plt.plot(n_evals, cv, '--', label="CV")
    plt.scatter(first_feas_evals, cv[k], color="red", label="First Feasible")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Constraint Violation (CV)")
    plt.legend()
    plt.show()

    import matplotlib.pyplot as plt
    from pymoo.performance_indicator.hv import Hypervolume

    # MODIFY - this is problem dependend
    ref_point = np.array([1.0, 1.0])

    # create the performance indicator object with reference point
    metric = Hypervolume(ref_point=ref_point, normalize=False)

    # calculate for each generation the HV metric
    hv = [metric.calc(f) for f in F]

    # visualze the convergence curve
    plt.plot(n_evals, hv, '-o', markersize=4, linewidth=2)
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.show()

    pf = problem.pareto_front(flatten=True, use_cache=False)

    import matplotlib.pyplot as plt
    from pymoo.performance_indicator.igd import IGD

    if pf is not None:
        # for this test problem no normalization for post prcessing is needed since similar scales
        normalize = False

        metric = IGD(pf=pf, normalize=normalize)

        # calculate for each generation the HV metric
        igd = [metric.calc(f) for f in F]

        # visualze the convergence curve
        plt.plot(n_evals, igd, '-o', markersize=4, linewidth=2, color="green")
        plt.yscale("log")  # enable log scale if desired
        plt.title("Convergence")
        plt.xlabel("Function Evaluations")
        plt.ylabel("IGD")
        plt.show()

    from pymoo.util.running_metric import RunningMetric

    running = RunningMetric(delta_gen=5,
                            n_plots=3,
                            only_if_n_plots=True,
                            key_press=False,
                            do_show=True)

    for algorithm in res.history[:15]:
        running.notify(algorithm)

    from pymoo.util.running_metric import RunningMetric

    running = RunningMetric(delta_gen=10,
                            n_plots=4,
                            only_if_n_plots=True,
                            key_press=False,
                            do_show=True)

    for algorithm in res.history:
        running.notify(algorithm)