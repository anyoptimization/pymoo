from pymoo.configuration import Configuration
from pymoo.model.crossover import Crossover
from pymoo.rand import random


class SimulatedBinaryCrossover(Crossover):
    def __init__(self, eta_xover=15, p_xover=0.9):
        super().__init__(2, 2)
        self.p_xover = p_xover
        self.eta_c = eta_xover

    def _do(self, p, parents, children, data=None):

        n = p.n_var

        if random.random() <= self.p_xover:

            for i in range(n):

                if random.random() <= 0.5:

                    if abs(parents[0, i] - parents[1, i]) > Configuration.EPS:

                        if parents[0, i] < parents[1, i]:
                            y1 = parents[0, i]
                            y2 = parents[1, i]
                        else:
                            y1 = parents[1, i]
                            y2 = parents[0, i]

                        yl = p.xl[i]
                        yu = p.xu[i]
                        rand = random.random()
                        beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.eta_c + 1.0))
                        if rand <= (1.0 / alpha):
                            betaq = pow((rand * alpha), (1.0 / (self.eta_c + 1.0)))

                        else:
                            betaq = pow((1.0 / (2.0 - rand * alpha)), (1.0 / (self.eta_c + 1.0)))

                        c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                        beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.eta_c + 1.0))
                        if rand <= (1.0 / alpha):
                            betaq = pow((rand * alpha), (1.0 / (self.eta_c + 1.0)))
                        else:
                            betaq = pow((1.0 / (2.0 - rand * alpha)), (1.0 / (self.eta_c + 1.0)))

                        c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                        if c1 < yl:
                            c1 = yl
                        if c2 < yl:
                            c2 = yl
                        if c1 > yu:
                            c1 = yu
                        if c2 > yu:
                            c2 = yu
                        if random.random() <= 0.5:
                            children[0, i] = c2
                            children[1, i] = c1

                        else:
                            children[0, i] = c1
                            children[1, i] = c2

                    else:
                        children[0, i] = parents[0, i]
                        children[1, i] = parents[1, i]

                else:
                    children[0, i] = parents[0, i]
                    children[1, i] = parents[1, i]

        else:
            children[:, :] = parents

        return children
