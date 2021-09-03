import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from prettytable import PrettyTable

from pymoo.algorithms.soo.nonconvex.cmaes import SimpleCMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.nelder_mead import NelderMead
from pymoo.algorithms.soo.nonconvex.pattern_search import PatternSearch
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.experimental.benchmarking.analyzer.convergence import ConvergenceAnalyzer
from pymoo.experimental.benchmarking.analyzer.groupby import GroupBy
from pymoo.experimental.benchmarking.analyzer.soo import SingleObjectiveAnalyzer
from pymoo.experimental.benchmarking.benchmark import Benchmark, DefaultWriter, DefaultLoader
from pymoo.experimental.benchmarking.recoder import DefaultSingleObjectiveRecorder
from pymoo.experimental.benchmarking.util import filter_by
from pymoo.factory import get_problem
from pymoo.util.termination.max_eval import MaximumFunctionCallTermination

if __name__ == "__main__":

    FOLDER = "/Users/blankjul/my_benchmark"

    recorder = DefaultSingleObjectiveRecorder()

    termination = MaximumFunctionCallTermination(100 * 1000)

    benchmark = Benchmark(n_runs=11, recorder=recorder)

    # benchmark.add_problem("sphere-10d", Sphere(n_var=10), ("n_evals", 1000))
    # benchmark.add_problem("himmelblau", Himmelblau(), ("n_evals", 400))
    # benchmark.add_problem("rosenbrock", Rosenbrock(), ("n_evals", 500))
    # benchmark.add_problem("ackley-10d", Ackley(n_var=10), ("n_evals", 1000))
    # benchmark.add_problem("rastrigin-5d", Rastrigin(n_var=5), ("n_evals", 1000))

    instance = 1
    n_evals = 1000
    for n_var in [10, 20, 40]:
        for function in range(1, 25):
            label = f"bbob-f{function:02d}-{instance}"
            benchmark.add_problem(label + "-" + str(n_var), get_problem(label, n_var=n_var), ("n_evals", n_evals))

    benchmark.add_algorithm("de", DE())
    benchmark.add_algorithm("ga", GA())
    benchmark.add_algorithm("pso", PSO())
    benchmark.add_algorithm("cmaes", SimpleCMAES())
    benchmark.add_algorithm("ps", PatternSearch())
    benchmark.add_algorithm("nm", NelderMead())

    loader = DefaultLoader(FOLDER)
    # loader = None

    writer = DefaultWriter(FOLDER)

    results = benchmark.run(writer=writer,
                            loader=loader,
                            run_if_loading_fails=True)

    # _ = SingleObjectiveAnalyzer().run(results, benchmark=benchmark, inplace=True)
    results = SingleObjectiveAnalyzer().run(results, benchmark=benchmark, inplace=False)

    attrs = {"fgap": ["np.median", "np.mean", "np.std", "collect"]}
    mean = GroupBy(attrs).run(results, group_by=["problem", "algorithm"])

    for scope, d in filter_by(mean, ["problem"], return_group=True):
        l = sorted(d, key=lambda x: x["fgap_median"])

        best = l[0]["fgap"]

        # _, pval = scipy.stats.levene(*[e["fgap"] for e in l])

        t = PrettyTable()
        t.title = scope["problem"]
        t.field_names = ['Algorithm', 'fgap_median', 'fgap_mean', 'fgap_std', 'shapiro', 'levene', 't-test', 'wilcoxon']

        for i, e in enumerate(l):
            f = e["fgap"]

            _, pval = scipy.stats.shapiro(f)
            shapiro = "*" if pval >= 0.01 else ""

            _, pval = scipy.stats.levene(best, f)
            levene = "* (%.3f)" % pval if pval >= 0.05 else ""

            _, pval = scipy.stats.ttest_ind(f, best, alternative="greater")
            ttest = "* (%.3f)" % pval if pval >= 0.05 else ""

            if len(best) == len(f):
                _, pval = scipy.stats.wilcoxon(f, best, zero_method="zsplit", alternative="greater")
                wilcoxon = "* (%.3f)" % pval if pval >= 0.05 else ""
            else:
                wilcoxon = "x"

            t.add_row(
                [e["algorithm"], "%.10f" % e["fgap_median"], "%.10f" % e["fgap_mean"], "%.10f" % e["fgap_std"], shapiro,
                 levene, ttest, wilcoxon])

        print(t)
        print()

    conv = ConvergenceAnalyzer().run(results, group_by=["problem", "algorithm"])

    attrs = {"conv_gap": "np.median", "n_evals": "first"}
    median = GroupBy(attrs).run(conv, group_by=["problem", "algorithm"])

    plot = False

    if plot:

        for scope, d in filter_by(median, ["problem"], return_group=True):

            plt.figure()
            plt.title(scope["problem"])
            plt.yscale("log")

            for entry in d:
                plt.plot(entry["n_evals"], entry["conv_gap_mean"], label=entry["algorithm"])

            plt.legend()
            plt.show()

    ranks = {}

    for scope, d in filter_by(mean, ["problem"], return_group=True):

        A = [e["algorithm"] for e in d]
        perf = np.array([e["fgap_median"] for e in d])
        rank = np.argsort(np.argsort(perf))

        for a, r in zip(A, rank):

            if a not in ranks:
                ranks[a] = []

            ranks[a].append(r + 1)

    cnt = np.array([len(v) for k, v in ranks.items()])
    assert cnt.min() == cnt.max(), "All algorithms must have been run on all problems!"
    k = cnt.min()

    R = np.column_stack([v for k, v in ranks.items()])
    R_total = R.mean(axis=0)

    t = PrettyTable(['problem'] + list(ranks.keys()))

    p = list(benchmark.problems.keys())

    for i, row in enumerate(R):
        t.add_row([p[i]] + list(row))

    t.add_row(["Total"] + [np.round(e, 3) for e in R_total])

    print(t)
