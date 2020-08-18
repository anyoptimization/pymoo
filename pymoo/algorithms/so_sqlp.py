from __future__ import division, print_function, absolute_import

import numpy as np
from numpy import (zeros, array, asfarray, concatenate)
from scipy.optimize._slsqp import slsqp

from pymoo.algorithms.so_local_search import LocalSearch
from pymoo.model.evaluator import set_cv
from pymoo.model.individual import Individual
from pymoo.model.population import Population
from pymoo.model.problem import Problem
from pymoo.util.approx_grad import GradientApproximation
from pymoo.util.display import SingleObjectiveDisplay


class SQLPDisplay(SingleObjectiveDisplay):

    def _do(self, problem, evaluator, algorithm):
        if algorithm.major:
            super()._do(problem, evaluator, algorithm)
            # self.output.append("iter", algorithm.D["majiter"])


class SQLP(LocalSearch):

    def __init__(self,
                 n_evals=1000,
                 ftol=1e-6,
                 display=SQLPDisplay(),
                 **kwargs):

        super().__init__(display=display, **kwargs)
        self.ftol = ftol
        self.n_evals = n_evals
        self.D = None
        self.pop = None
        self.majiter_prev = 1

        self.args = ["m", "meq", "X", "xl", "xu", "F", "G", "dF", "dG", "acc", "majiter", "mode", "w", "jw",
                     "alpha", "f0", "gs", "h1", "h2", "h3", "h4", "t", "t0", "tol",
                     "iexact", "incons", "ireset", "itermx", "line", "n1", "n2", "n3"]

    def setup(self, problem, **kwargs):
        super().setup(problem, **kwargs)

        n, meq, mieq = problem.n_var, 0, problem.n_constr
        m = meq + mieq
        la = max(1, m)

        n1 = n + 1
        mineq = m - meq + n1 + n1
        len_w = (3 * n1 + m) * (n1 + 1) + (n1 - meq + 1) * (mineq + 2) + 2 * mineq + (n1 + mineq) * (n1 - meq) \
                + 2 * meq + n1 + ((n + 1) * n) // 2 + 2 * m + 3 * n + 3 * n1 + 1
        len_jw = mineq

        self.D = dict(
            X=None,
            n=n,
            meq=meq,
            mieq=mieq,
            la=la,
            m=m,
            xl=problem.xl.astype(np.float),
            xu=problem.xu.astype(np.float),
            mode=array(0, int),
            acc=array(self.ftol, float),
            majiter=array(self.n_evals, int),
            majiter_prev=0,
            alpha=array(0, float),
            f0=array(0, float),
            gs=array(0, float),
            h1=array(0, float),
            h2=array(0, float),
            h3=array(0, float),
            h4=array(0, float),
            t=array(0, float),
            t0=array(0, float),
            tol=array(0, float),
            iexact=array(0, int),
            incons=array(0, int),
            ireset=array(0, int),
            itermx=array(0, int),
            line=array(0, int),
            n1=array(0, int),
            n2=array(0, int),
            n3=array(0, int),
            w=zeros(len_w),
            jw=zeros(len_jw)
        )

    def _initialize(self):
        super()._initialize()

        # Clip initial guess to bounds (SLSQP may fail with bounds-infeasible initial point)
        x = asfarray(self.x0.X.flatten())
        xl, xu = self.problem.bounds()
        have_bound = np.isfinite(xl)
        x[have_bound] = np.clip(x[have_bound], xl[have_bound], np.inf)
        have_bound = np.isfinite(xu)
        x[have_bound] = np.clip(x[have_bound], -np.inf, xu[have_bound])
        self.D["X"] = x

        self.pop = Population()
        self._eval_obj()
        self._eval_grad()
        self._update()
        self._call()

        self.major = True

    def _update(self):
        D = self.D
        ind = Individual(X=np.copy(D["X"]), F=np.copy(D["F"]), G=np.copy(-D["G"]))
        pop = Population.merge(self.pop, Population.create(ind))
        set_cv(pop)
        self.pop = pop

    def _call(self):
        # args = np.load("/Users/blankjul/workspace/pymoo/pymoo/algorithms/data_constr.npy", allow_pickle=True)
        #
        # pymoo = [self.D[e] for e in self.args]
        # for k in range(len(self.args)):
        #     orig = tuple(args)[k]
        #     _pymoo = pymoo[k]
        #     if isinstance(orig, np.ndarray):
        #         print(orig.dtype, _pymoo.dtype)
        #         print(orig.shape, _pymoo.shape)
        #     print(type(orig), type(_pymoo))
        #     print(orig)
        #     print(_pymoo)
        #
        #     print()
        #
        # print("-------------------------------------------------")

        slsqp(*[self.D[e] for e in self.args])

    def _step(self):
        self._eval_obj()
        self._update()

        self._call()

        if self.D["mode"] == -1:
            self._eval_grad()
            self._call()

    def _eval_obj(self):
        D = self.D
        D["F"], _, G = self.evaluator.eval(self.problem, D["X"])
        D["G"] = - G if self.problem.n_constr > 0 else np.zeros(0)

    def _eval_grad(self):
        D = self.D

        # dF, dG = self.problem.evaluate(D["X"], return_values_of=["dF", "dG"])
        # if dF is None or dG is None:
        #     dF, dG = GradientApproximation(self.problem, evaluator=self.evaluator).do(Individual(X=D["X"]))

        dF, dG = GradientApproximation(self.problem, evaluator=self.evaluator).do(Individual(X=D["X"]))

        D["dF"] = concatenate([dF[0], [0]])
        if self.problem.n_constr > 0:
            D["dG"] = - np.column_stack([dG, zeros(self.problem.n_constr)])
        else:
            D["dG"] = zeros((1, self.problem.n_var + 1))

    def _next(self):
        self._step()

        majiter = int(self.D["majiter"])
        if abs(self.D["mode"]) != 1:
            self.termination.force_termination = True

        self.major = self.majiter_prev < majiter
        self.majiter_prev = majiter
