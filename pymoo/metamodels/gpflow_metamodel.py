import gpflow
import numpy
import scipy
from gpflow.mean_functions import Linear

from metamodels.metamodel import MetaModel
from model.problem import Problem
from operators.lhs_factory import LHS


class GPFlowMetamodel(MetaModel):
    def _get_parameter(self, d={}):
        n_var = d['n_var']

        return [gpflow.kernels.Exponential(n_var)] #, gpflow.kernels.Polynomial(n_var, ARD=True, degree=5), gpflow.kernels.RBF(n_var, ARD=True)]

        k = gpflow.kernels.Constant(n_var)

        for kernel_type in [gpflow.kernels.RBF, gpflow.kernels.Matern12, gpflow.kernels.Linear]:
            for i in range(n_var):
                k *= kernel_type(input_dim=1, active_dims=[i])

        return [k]

    def _predict(self, metamodel, X):
        mean, cov = metamodel.predict_y(X)

        mean = mean.T
        cov = cov.T
        if mean.shape[0] == 1:
            mean = mean[0]
        if cov.shape[0] == 1:
            cov = cov[0]
        return mean, cov

    def _create_and_fit(self, parameter, X, F, expensive=False):
        m = gpflow.gpr.GPR(X, numpy.array([F]).T, kern=parameter, mean_function=Linear(numpy.ones((X.shape[1], 1)), numpy.ones((1, 1))))

        def minimize(fun, x0):
            n_var = len(x0)
            p = Problem(n_var=n_var, n_obj=1, n_constr=0, xl=-10, xu=10, func=None)

            def evaluate(x):
                f = numpy.zeros((x.shape[0], 1))
                g = numpy.zeros((x.shape[0], 0))

                if x.ndim == 1:
                    x = numpy.array([x])
                for i in range(x.shape[0]):
                    f[i, :] = fun(x[i, :])[0]
                return f, g

            p.evaluate = evaluate


            #X,F,G = NSGA(pop_size=300).solve(p, 60000)

            #X, F, G = solve_by_de(p)

            print(X)
            print(F)

            return X[0,:]


        res = m.optimize()

        print("GPFLOW")
        print(res.x)
        print(res.fun)

        print("DE:")
        #m.optimize(method=


        def minimize2(fun, x0):
            n_var = len(x0)
            val = scipy.optimize.differential_evolution(lambda x: fun(x)[0], [(-10,10) for _ in range(n_var)],
                                                        popsize=40, mutation=(0.7, 1), recombination=0.3,)
            print(val.fun)
            print(val.x)

            return val.x
            #p = Problem(n_var=n_var, n_obj=1, n_constr=0, xl=-10, xu=10, func=None)

        def minimize3(fun, x0):
            n_var = len(x0)
            n_restarts = 30

            initial_points = LHS().sample(n_restarts, -10 * numpy.ones(n_var), 10 * numpy.ones(n_var))

            #initial__point = [x0]
            #for _ in range(n_restarts-1):
            #    initial__point.append(numpy.random.random(n_var))

            results = []
            for p in initial_points:
                result = scipy.optimize.minimize(fun=fun,
                                  x0=p,
                                  method='L-BFGS-B',
                                  jac=True,
                                  tol=None,
                                  callback=None)
                results.append(result)

            idx = numpy.argmin([e.fun for e in results])
            result = results[idx]
            print(result.x)
            print(result.fun)
            return result.x

        if expensive:
            m.optimize(method=minimize3)

        #print(m)
        print("---------")
        return m
