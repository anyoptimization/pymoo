from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.reference_direction import get_partition_closest_to_points, ReferenceDirectionFactory


class ReferenceDirectionGA(ReferenceDirectionFactory):

    def __init__(self,
                 n_dim,
                 n_points,
                 fun,
                 pop_size=20,
                 n_gen=200,
                 verbose=False,
                 **kwargs):

        super().__init__(n_dim, **kwargs)

        self.n_points = n_points
        self.pop_size = pop_size
        self.n_gen = n_gen

        self.fun = fun
        self.verbose = verbose

    def _do(self, random_state=None):
        pop_size, n_gen = self.pop_size, self.n_gen
        n_points, n_dim, = self.n_points, self.n_dim
        fun = self.fun

        class MyProblem(Problem):

            def __init__(self):
                self.n_points = n_points
                self.n_dim = n_dim
                self.n_partitions = get_partition_closest_to_points(n_points, n_dim)

                super().__init__(n_var=n_points * n_dim,
                                 n_obj=1,
                                 xl=0.0,
                                 xu=1.0,
                                 elementwise_evaluation=True)

            def get_points(self, x):
                _x = x.reshape((self.n_points, self.n_dim)) ** 2
                _x = _x / _x.sum(axis=1)[:, None]
                return _x

            def _evaluate(self, x, out, *args, **kwargs):
                out["F"] = fun(self.get_points(x))

        problem = MyProblem()

        algorithm = GA(pop_size=pop_size, eliminate_duplicates=True)

        res = minimize(problem,
                       algorithm,
                       termination=('n_gen', n_gen),
                       random_state=random_state,
                       verbose=True)

        ref_dirs = problem.get_points(res.X)
        return ref_dirs
