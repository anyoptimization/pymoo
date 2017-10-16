import pygmo

from vendor.pygmo_algorithm import PygmoAlgorithm


def solve_by_moad(problem, pop_size=100, gen=100):
    a = PygmoAlgorithm(pygmo.moead(gen=gen, weight_generation="grid", decomposition="tchebycheff",
                                   neighbours=5, CR=1, F=0.5, eta_m=20, realb=0.9, limit=2, preserve_diversity=True)
                       , pop_size=pop_size)
    return a.solve(problem, None, None)