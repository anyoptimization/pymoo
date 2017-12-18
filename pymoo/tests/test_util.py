from pymoo.model.individual import Individual


def make_individual(f, g):
    ind = Individual()
    ind.f = f
    ind.g = g
    return ind
