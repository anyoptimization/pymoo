import numpy as np
import optuna

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.factory import get_problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize


def objective(trial):
    problem = get_problem("ackley", n_var=10)
    n_var = problem.n_var

    xprob = trial.suggest_float('xprob', 0.75, 1.0)
    xeta = trial.suggest_float('xeta', 3.0, 30.0)
    xprob_var_var = trial.suggest_float('xprob_per_var', 1 / n_var, 0.75)
    xprob_binomial = trial.suggest_float('prob_bin', 1 / n_var, 0.75)

    crossover = SBX(prob=xprob, eta=xeta, prob_per_variable=xprob_var_var, prob_binomial=xprob_binomial)

    mprob = trial.suggest_float('mprob', 0.75, 1.0)
    meta = trial.suggest_float('meta', 3.0, 30.0)
    mprob_var_var = trial.suggest_float('mprob_per_var', 1 / n_var, 0.5)
    mutation = PM(prob=mprob, eta=meta, prob_per_variable=mprob_var_var)

    algorithm = GA(crossover=crossover, mutation=mutation)

    f = []
    for k in range(5):
        res = minimize(problem, algorithm, ("n_gen", 300), seed=k)
        f.append(res.F.min())

    return np.array(f).mean()


study = optuna.create_study()
study.optimize(objective, n_trials=30)

print(study.best_params)
