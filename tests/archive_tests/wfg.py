import numpy as np

from pymoo.model.problem import Problem


class WFG(Problem):
    def __init__(self, name, n_var, n_obj, k=None):
        Problem.__init__(self)
        self.n_obj = n_obj
        self.n_var = n_var
        self.k = 2 * (self.n_obj - 1) if k is None else k
        self.func = self._evaluate
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)

        # function used to evaluate
        self.func = self._evaluate

        # the function pointing to the optproblems implementation
        exec('import optproblems.wfg')
        clazz, = eval('optproblems.wfg.%s' % name),
        self._func = clazz(num_objectives=self.n_obj, num_variables=self.n_var, k=self.k)

    def _evaluate(self, x, f):
        for n in range(len(x)):
            z = x[n, :]
            f[n, :] = self._func(z)

    def pareto_front(self):
        n_optimal_solution = 1000
        pf = np.zeros((n_optimal_solution, self.n_obj))

        s = self._func.get_optimal_solutions(n_optimal_solution)
        for n in range(len(s)):
            pf[n, :] = self._func(s[n].phenome)
        return pf


class WFG1(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(self.__class__.__name__, n_var, n_obj, k=k)


class WFG2(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(self.__class__.__name__, n_var, n_obj, k=k)


class WFG3(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(self.__class__.__name__, n_var, n_obj, k=k)


class WFG4(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(self.__class__.__name__, n_var, n_obj, k=k)


class WFG5(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(self.__class__.__name__, n_var, n_obj, k=k)


class WFG6(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(self.__class__.__name__, n_var, n_obj, k=k)


class WFG7(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(self.__class__.__name__, n_var, n_obj, k=k)


class WFG8(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(self.__class__.__name__, n_var, n_obj, k=k)


class WFG9(WFG):
    def __init__(self, n_var=10, n_obj=3, k=None):
        super().__init__(self.__class__.__name__, n_var, n_obj, k=k)


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    problem = WFG1(n_var=12, n_obj=3, k=4)
    pf = problem.pareto_front()

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2])
    plt.show()
