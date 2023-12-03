from pymoo.core.solution import merge, Solution


def sort_by_fitness(sols):
    feas = sols.get("feas")
    return merge(sorted(sols[feas], key=lambda s: s.f), sorted(sols[~feas], key=lambda s: s.cv))


def get_fittest(sols):
    if len(sols) > 0:
        return sort_by_fitness(sols)[0]

def is_better(a: Solution, b: Solution):
    if a.feas and b.feas:
        return a.f < b.f
    else:
        return a.cv < b.cv
