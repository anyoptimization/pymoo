import operator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.util.remote import Remote

from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

file = Remote.get_instance().load("examples", "portfolio_allocation.csv", to=None)

df = pd.read_csv(file, parse_dates=True, index_col="date")

returns = df.pct_change().dropna(how="all")
mu = (1 + returns).prod() ** (252 / returns.count()) - 1
cov = returns.cov() * 252

mu, cov = mu.to_numpy(), cov.to_numpy()


class PortfolioProblem(ElementwiseProblem):

    def __init__(self, mu, cov, risk_free_rate=0.02, **kwargs):
        super().__init__(n_var=len(df.columns), n_obj=2, xl=0.0, xu=1.0, **kwargs)
        self.mu = mu
        self.cov = cov
        self.risk_free_rate = risk_free_rate

    def _evaluate(self, x, out, *args, **kwargs):
        exp_return = x @ self.mu
        exp_risk = np.sqrt(x.T @ self.cov @ x)
        sharpe = (exp_return - self.risk_free_rate) / exp_risk

        out["F"] = [exp_risk, -exp_return]
        out["sharpe"] = sharpe


class PortfolioSampling(FloatRandomSampling):

    def __init__(self, mu, cov) -> None:
        super().__init__()
        self.mu = mu
        self.cov = cov

    def _do(self, problem, n_samples, **kwargs):
        X = super()._do(problem, n_samples, **kwargs)

        n = len(mu)
        n_biased = min(n, n_samples // 2)

        order_by_ret = (-self.mu).argsort()
        order_by_cov = (self.cov.diagonal()).argsort()
        order = np.stack([order_by_ret, order_by_cov]).min(axis=0)

        X[:n_biased] = np.eye(n)[order][:n_biased]

        return X


class PortfolioRepair(Repair):

    def _do(self, problem, X, **kwargs):
        X[X < 1e-3] = 0
        return X / X.sum(axis=1, keepdims=True)


problem = PortfolioProblem(mu, cov)

algorithm = SMSEMOA(sampling=PortfolioSampling(mu, cov), repair=PortfolioRepair())

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True)

X, F, sharpe = res.opt.get("X", "F", "sharpe")
F = F * [1, -1]
max_sharpe = sharpe.argmax()

allocation = {name: w for name, w in zip(df.columns, X[max_sharpe])}
allocation = sorted(allocation.items(), key=operator.itemgetter(1), reverse=True)

print("Allocation With Best Sharpe")
for name, w in allocation:
    print(f"{name:<5} {w}")

plt.scatter(F[:, 0], F[:, 1], facecolor="none", edgecolors="blue", alpha=0.5, label="Pareto-Optimal Portfolio")
plt.scatter(cov.diagonal() ** 0.5, mu, facecolor="none", edgecolors="black", s=30, label="Asset")
plt.scatter(F[max_sharpe, 0], F[max_sharpe, 1], marker="x", s=100, color="red", label="Max Sharpe Portfolio")
plt.legend()
plt.show()
