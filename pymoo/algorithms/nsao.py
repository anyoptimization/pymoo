import copy

from algorithms.nsga import NSGA
from metamodels.metamodel import MetaModel
from model.algorithm import Algorithm
from model.individual import Individual
from operators.random_spare_factory import RandomSpareFactory
from util.misc import get_x, get_f, evaluate
import matplotlib.pyplot as plt
import numpy as np

from util.non_dominated_rank import NonDominatedRank


class NSAO(Algorithm):
    def __init__(self,
                 initial_DOE=0.5,
                 DOE_per_epoch=4,
                 factory=RandomSpareFactory()  # factory for the initiation population
                 ):
        self.initial_DOE = initial_DOE
        self.DOE_per_epoch = DOE_per_epoch
        self.factory = factory

    def solve_(self, problem, evaluator):
        # initial DOE
        pop = [Individual(x) for x in
               self.factory.sample(int(evaluator.n_eval * self.initial_DOE), problem.xl, problem.xu)]
        evaluate(evaluator, problem, pop)

        while evaluator.has_next():
            # learn a metamodel on the data
            meta_model = MetaModel()
            meta_model.fit(get_x(pop), get_f(pop))

            # optimize the problem on the metamodel
            meta_model_problem = copy.deepcopy(problem)
            meta_model_problem.n_constr = 1

            def evaluate_(x, f, g, pop=pop):
                f[:] = meta_model.predict(x)
                min_dist = np.min([np.linalg.norm(x-ind.x) for ind in pop])
                g[0] = min_dist - 1.0

            meta_model_problem.func = evaluate_

            predicted_front = NSGA().solve(meta_model_problem, 10000)
            predicted_front = [predicted_front[i] for i in NonDominatedRank.calc_as_fronts(predicted_front)[0]]

            selected = self.select_from_front(predicted_front)
            print get_f(selected)
            evaluate(evaluator, problem, selected)
            print get_f(selected)
            pop.extend(selected)


            f = get_f(predicted_front)
            #plt.scatter(f[:, 0], f[:, 1])
            #plt.show()

        front = [pop[i] for i in NonDominatedRank.calc_as_fronts(pop)[0]]
        f = get_f(front)
        plt.scatter(f[:, 0], f[:, 1])
        plt.show()

        print 'Done'

    def select_from_front(self, front):
        return front[:self.DOE_per_epoch]

