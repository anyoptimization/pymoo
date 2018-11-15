"""
This is the experiment for nsga2.
"""
import os
import pickle

from experiments.experiment_nsga3 import setup
from pymoo.experimental.emo_new.keep_extreme import ReferenceDirectionSurvivalKeepExtreme
from pymoo.experimental.emo_new.nsga3_pbi import ReferenceDirectionSurvivalPBI
from pymoo.algorithms.nsga3 import NSGA3
from pymoo.model.termination import MaximumGenerationTermination
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation

if __name__ == '__main__':

    n_runs = 30
    # problems = ['zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6']
    problems = setup.keys()

    for e in problems:

        s = setup[e]
        problem = s['problem']

        algorithm = NSGA3(s['ref_dirs'], pop_size=s['pop_size'])
        algorithm.crossover = SimulatedBinaryCrossover(0.9, 15)
        algorithm.mutation = PolynomialMutation(20)
        algorithm.survival = ReferenceDirectionSurvivalPBI(s['ref_dirs'])

        s = setup[e]
        problem = s['problem']

        for run in range(n_runs):
            data = {
                'problem': problem,
                'algorithm': algorithm,
                'seed': run,
                'termination': MaximumGenerationTermination(s['termination'][1]),
                'out': "%s/pynsga3_%s_%s.out" % (e.replace("_", "/"), e, (run + 1)),
            }

            folder = "pynsga3-pbi"
            os.makedirs(folder, exist_ok=True)
            fname = "pynsga3_%s_%s.run" % (e, (run + 1))
            with open(os.path.join(folder, fname), 'wb') as f:
                pickle.dump(data, f)
