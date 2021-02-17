from pymoo.algorithms.soo.nonconvex.pso import PSO, PSOAnimation
from pymoo.factory import Rastrigin
from pymoo.optimize import minimize
from pyrecorder.video import Video

problem = Rastrigin()

algorithm = PSO()

res = minimize(problem,
               algorithm,
               callback=PSOAnimation(fname="pso.mp4", nth_gen=5),
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
