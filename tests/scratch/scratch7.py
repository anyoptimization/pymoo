import autograd.numpy as anp
from pymoo.model.problem import Problem
from pymoo.visualization.scatter import Scatter


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=2,
                         xl=anp.array([-2, -2]),
                         xu=anp.array([2, 2]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0] ** 2 + x[:, 1] ** 2
        f2 = (x[:, 0] - 1) ** 2 + x[:, 1] ** 2

        g1 = 2 * (x[:, 0] - 0.1) * (x[:, 0] - 0.9) / 0.18
        g2 = - 20 * (x[:, 0] - 0.4) * (x[:, 0] - 0.6) / 4.8

        out["F"] = anp.column_stack([f1, f2])
        out["G"] = anp.column_stack([g1, g2])


from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation

algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=20),
    eliminate_duplicates=True
)


from pymoo.optimize import minimize

res = minimize(MyProblem(),
               algorithm,
               ('n_gen', 40),
               seed=1,
               verbose=True)

plot = Scatter()
plot.add(res.F, color="red")
plot.show()


plot = Scatter()
plot.add(res.X, color="red")
plot.do()
import matplotlib.pyplot as plt
plt.ylim(-2,2)
plt.show()


