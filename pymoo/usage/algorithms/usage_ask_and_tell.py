from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.performance_indicator.igd import IGD

from pymoo.util.ask_and_tell import AskAndTell
from pymoo.util.termination.no_termination import NoTermination

problem = get_problem("zdt1")
pf = problem.pareto_front()

algorithm = NSGA2().setup(problem, termination=NoTermination(), verbose=False)

interface = AskAndTell(algorithm)

for k in range(200):
    pop = interface.ask()

    algorithm.evaluator.eval(problem, pop)

    interface.tell(pop)
    print(k + 1, IGD(pf).calc(algorithm.opt.get("F")))

print(algorithm.opt.get("F"))
