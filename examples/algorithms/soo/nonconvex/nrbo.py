import numpy as np

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.nrbo import NRBO
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.pso_ep import EPPSO
from pymoo.operators.sampling.lhs import LHS
from pymoo.problems.single import Ackley
from pymoo.visualization.matplotlib import plt

prob = Ackley(n_var=50)
seed = 1
pop_size = 50
max_iter = 100

# ------ nrbo ------
alg = NRBO(pop_size=pop_size, deciding_factor=0.6, sampling=LHS(), max_iteration=max_iter)
alg.setup(problem=prob, seed=seed)
F_hist_nrbo = []
while alg.has_next():
    alg.next()
    pop = alg.pop.get("F")
    pop_size = len(pop)
    F_best = alg.opt.get("F")
    F_best = np.tile(F_best, (pop_size, 1))
    F_hist_nrbo.append(F_best)
F_hist_nrbo = np.vstack(F_hist_nrbo)
res = alg.result()
print(res.F)
print(res.X)

# ------ ga ------
alg = GA(pop_size=pop_size, sampling=LHS())
alg.setup(problem=prob, termination=("n_gen", max_iter), seed=seed)
F_hist_ga = []
while alg.has_next():
    alg.next()
    pop = alg.pop.get("F")
    pop_size = len(pop)
    F_best = alg.opt.get("F")
    F_best = np.tile(F_best, (pop_size, 1))
    F_hist_ga.append(F_best)
F_hist_ga = np.vstack(F_hist_ga)
res = alg.result()
print(res.F)

# ------ de ------
alg = DE(pop_size=pop_size, sampling=LHS())
alg.setup(problem=prob, termination=("n_gen", max_iter), seed=seed)
F_hist_de = []
while alg.has_next():
    alg.next()
    pop = alg.pop.get("F")
    pop_size = len(pop)
    F_best = alg.opt.get("F")
    F_best = np.tile(F_best, (pop_size, 1))
    F_hist_de.append(F_best)
F_hist_de = np.vstack(F_hist_de)
res = alg.result()
print(res.F)

# ------ PSO ------
alg = PSO(pop_size=pop_size, sampling=LHS())
alg.setup(problem=prob, termination=("n_gen", max_iter), seed=seed)
F_hist_pso = []
while alg.has_next():
    alg.next()
    pop = alg.pop.get("F")
    pop_size = len(pop)
    F_best = alg.opt.get("F")
    F_best = np.tile(F_best, (pop_size, 1))
    F_hist_pso.append(F_best)
F_hist_pso = np.vstack(F_hist_pso)
res = alg.result()
print(res.F)

# ------ EPPSO ------
alg = EPPSO(pop_size=pop_size, sampling=LHS())
alg.setup(problem=prob, termination=("n_gen", max_iter), seed=seed)
F_hist_eppso = []
while alg.has_next():
    alg.next()
    pop = alg.pop.get("F")
    pop_size = len(pop)
    F_best = alg.opt.get("F")
    F_best = np.tile(F_best, (pop_size, 1))
    F_hist_eppso.append(F_best)
F_hist_eppso = np.vstack(F_hist_eppso)
res = alg.result()
print(res.F)

fig = plt.figure(0, figsize=(10.0, 8.0))
ax = fig.add_subplot(111)
ax.plot(F_hist_nrbo, label="nrbo")
ax.plot(F_hist_ga, label="ga")
ax.plot(F_hist_de, label="de")
ax.plot(F_hist_pso, label="pso")
ax.plot(F_hist_eppso, label="eppso")
ax.legend()
ax.set_title("Ackley with 50 dim")
plt.show()
# fig.savefig("result.png",dpi=300)
