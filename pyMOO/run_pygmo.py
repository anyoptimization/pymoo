import matplotlib.pyplot as plt
import pygmo as pg

if __name__ == '__main__':

    prob = pg.problem(pg.zdt(prob_id=3))
    algo = pg.nsga2(100)
    archi = pg.archipelago(1, algo=algo, prob=prob, pop_size=100)
    archi.evolve(1)
    archi.wait()
    res = [isl.get_population() for isl in archi]
    f = res[0].get_f()

    plt.scatter(f[:, 0], f[:, 1])
