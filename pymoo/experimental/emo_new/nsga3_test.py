from pymoo.experimental.emo_new.keep_extreme import ReferenceDirectionSurvivalKeepExtreme
from pymoo.experimental.emo_new.nsga3_pbi import ReferenceDirectionSurvivalPBI
from pymoo.algorithms.nsga3 import NSGA3
from pymoo.model.termination import MaximumGenerationTermination
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.util import plotting
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymop.factory import get_problem

problem = get_problem("dtlz1", n_var=7, n_obj=3)
#problem = get_problem("dtlz4", n_var=12, n_obj=3)
# create the reference directions to be used for the optimization
ref_dirs = UniformReferenceDirectionFactory(3, n_points=91).do()

# create the pareto front for the given reference lines
pf = problem.pareto_front(ref_dirs)

algorithm = NSGA3(ref_dirs)
algorithm.crossover = SimulatedBinaryCrossover(0.9, 15)
algorithm.mutation = PolynomialMutation(20)
algorithm.survival = ReferenceDirectionSurvivalPBI(ref_dirs)
termination = MaximumGenerationTermination(400)

res = algorithm.solve(problem, termination, seed=1, disp=True, pf = pf)

plotting.plot(res.F)
