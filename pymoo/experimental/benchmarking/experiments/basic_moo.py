import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from prettytable import PrettyTable

from pymoo.algorithms.moo.gde3 import GDE3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.cmaes import SimpleCMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.nelder_mead import NelderMead
from pymoo.algorithms.soo.nonconvex.pattern_search import PatternSearch
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.experimental.benchmarking.analyzer.convergence import ConvergenceAnalyzer
from pymoo.experimental.benchmarking.analyzer.groupby import GroupBy
from pymoo.experimental.benchmarking.analyzer.moo import MultiObjectiveAnalyzer
from pymoo.experimental.benchmarking.analyzer.others import MultiObjectiveConvergenceAnalyzer
from pymoo.experimental.benchmarking.analyzer.soo import SingleObjectiveAnalyzer
from pymoo.experimental.benchmarking.benchmark import Benchmark, DefaultWriter, DefaultLoader
from pymoo.experimental.benchmarking.recoder import DefaultSingleObjectiveRecorder, DefaultMultiObjectiveRecorder
from pymoo.experimental.benchmarking.util import filter_by
from pymoo.factory import get_problem, ZDT1, ZDT2, ZDT3
from pymoo.util.termination.max_eval import MaximumFunctionCallTermination

if __name__ == "__main__":
    FOLDER = "/Users/blankjul/moo_benchmark"

    recorder = None
    recorder = DefaultMultiObjectiveRecorder()

    benchmark = Benchmark(n_runs=11, recorder=recorder)

    benchmark.add_problem("zdt1", ZDT1(), termination=("n_gen", 200))
    # benchmark.add_problem("zdt2", ZDT2(), termination=("n_gen", 200))
    # benchmark.add_problem("zdt3", ZDT3(), termination=("n_gen", 200))

    benchmark.add_algorithm("nsga2", NSGA2())
    benchmark.add_algorithm("gde3", GDE3())

    loader = DefaultLoader(FOLDER)
    writer = DefaultWriter(FOLDER)

    results = benchmark.run(writer=writer,
                            loader=loader,
                            run_if_loading_fails=True)

    # set the igd values for each of the problems
    MultiObjectiveAnalyzer().run(results, benchmark=benchmark, inplace=True)

    # now aggregate all the runs to have some representative values
    attrs = [("igd", np.array, "igd"),
             ("igd", np.mean, "avg"),
             ("igd", np.std, "std")]

    igd = GroupBy(attrs).run(results, group_by=["problem", "algorithm"])

    for scope, d in filter_by(igd, ["problem"], return_group=True):

        # find the best algorithm for this problem
        l = sorted(d, key=lambda x: x["avg"])
        best = l[0]["igd"]

        t = PrettyTable()
        t.title = scope["problem"]
        t.field_names = ['Algorithm', 'avg', 'std', 'shapiro', 'levene', 't-test', 'wilcoxon']

        for i, e in enumerate(l):
            f = e["igd"]

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

            t.add_row([e["algorithm"], "%.10f" % e["avg"], "%.10f" % e["std"], shapiro, levene, ttest, wilcoxon])

        print(t)
        print()


