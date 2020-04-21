from pymoo.algorithms.so_pso import PSO
from pymoo.factory import get_problem, Sphere
from pymoo.optimize import minimize

problem = get_problem("rastrigin", n_var=4)

algorithm = PSO()

ret = minimize(problem,
               algorithm,
               seed=1,
               save_history=True,
               verbose=True)

print(ret.X)
print("Optimal:", problem.pareto_front()[0])
print(ret.F)


if False:

    with Video(File("pso.mp4")) as vid:
        for algorithm in ret.history:

            if algorithm.n_gen % 10 == 0:
                fl = FitnessLandscape(problem,
                                      _type="contour",
                                      kwargs_contour=dict(linestyles="solid", offset=-1, alpha=0.4))
                fl.do()

                X = algorithm.pop.get("X")
                plt.scatter(X[:, 0], X[:, 1], marker="x")

                X = algorithm.opt.get("X")
                plt.scatter(X[:, 0], X[:, 1], marker="x", s=20, color="red")

                # plt.legend()
                plt.title(algorithm.n_gen)

                vid.record()
