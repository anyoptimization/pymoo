import time

from pymoo.algorithms.nsga3 import NSGA3
from pymoo.model.termination import MaximumGenerationTermination
from pymoo.util.plotting import plot, animate
from pymoo.util.reference_direction import UniformReferenceDirectionFactory, MultiLayerReferenceDirectionFactory
from pymop import DTLZ3
from pymop.problems.dtlz import DTLZ1


def run():
    start_time = time.time()


    name = "3obj"

    if name == "3obj":
        ref_dirs = UniformReferenceDirectionFactory(3, n_partitions=12, scaling=1.0).do()
        problem = DTLZ3(n_var=7, n_obj=3)
        problem.n_pareto_points = 92
        pf = problem.pareto_front()

    elif name == "15obj":
        n_obj = 15

        ref_dirs = MultiLayerReferenceDirectionFactory([
            UniformReferenceDirectionFactory(15, n_partitions=2, scaling=1.0),
            UniformReferenceDirectionFactory(15, n_partitions=1, scaling=0.5)]).do()

        n_var = n_obj + 5 - 1
        problem = DTLZ1(n_var=n_var, n_obj=n_obj)
        problem.n_pareto_points = ref_dirs.shape[0]
        pf = problem.pareto_front()

    else:
        print("not konwn")

    """
    n_obj = 15

    ref_dirs = MultiLayerReferenceDirectionFactory([
        UniformReferenceDirectionFactory(15, n_partitions=2, scaling=1.0),
        UniformReferenceDirectionFactory(15, n_partitions=1, scaling=0.5)]).do()

    n_var = n_obj + 5 - 1
    problem = DTLZ1(n_var=n_var, n_obj=n_obj)
    problem.n_pareto_points = ref_dirs.shape[0]
    pf = problem.pareto_front()
    """


    """
    ref_dirs = UniformReferenceDirectionFactory(3, n_partitions=12, scaling=1.0).do()
    problem = DTLZ2(n_var=12, n_obj=3)
    problem.n_pareto_points = 92
    pf = problem.pareto_front()
    """


    """
    ref_dirs = MultiLayerReferenceDirectionFactory([
        UniformReferenceDirectionFactory(5, n_partitions=3, scaling=1.0),
        UniformReferenceDirectionFactory(5, n_partitions=2, scaling=0.5)]).do()

    problem = ScaledProblem(DTLZ2(n_var=13, n_obj=5), 10)
    problem.problem.n_pareto_points = 92
    pf = problem.problem.pareto_front() * problem.get_scale(5, 10)
    """

    algorithm = NSGA3(ref_dirs)
    #algorithm.survival = ProposeReferenceLineSurvival(ref_dirs)

    res = algorithm.solve(problem,
                          termination=MaximumGenerationTermination(450),
                          seed=0,
                          pf=pf,
                          save_history=True,
                          disp=True)

    X, F, history = res['X'], res['F'], res['history']

    print("--- %s seconds ---" % (time.time() - start_time))
    print(algorithm.survival.nadir_point)

    scatter_plot = True
    save_animation = False

    if scatter_plot:
        plot(F, problem)

    if save_animation:
        animate('%s.mp4' % problem.name(), history, problem)


if __name__ == '__main__':
    run()
