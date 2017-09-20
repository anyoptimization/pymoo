import matplotlib.pyplot as plt
import os

from moo.nsga.nsga import NSGA
from moo.operators.polynomial_mutation import PolynomialMutation
from moo.operators.random_factory import RandomFactory
from moo.operators.simulated_binary_crossover import SimulatedBinaryCrossover
from moo.problems.kursawe import Kursawe
from moo.problems.zdt import ZDT1, ZDT2, ZDT3



def write_final_pop_obj(pop, run):
    f_name = os.path.join('..', 'results', problem.__class__.__name__ + '_RUN' + str(run) + str('.out'))
    f = open(f_name, 'w')
    for ind in pop:
        f.write('%f \t %f\n' % (ind.f[0], ind.f[1]))
    f.close()



if __name__ == '__main__':
    problem = ZDT3()

    #print os.path.abspath(__file__)

    nsga = NSGA()
    nsga.factory = RandomFactory(problem.xl, problem.xu)
    nsga.crossover = SimulatedBinaryCrossover(problem.xl, problem.xu)
    nsga.mutation = PolynomialMutation(problem.xl, problem.xu)
    #nsga.callback = lambda x: xxxxx(x)

    pop = nsga.solve(problem, 200, 1)

    x = [pop[i].f[0] for i in range(len(pop))]
    y = [pop[i].f[1] for i in range(len(pop))]
    plt.scatter(x, y)
    plt.show()

    write_final_pop_obj(pop,1)

