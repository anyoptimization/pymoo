def get_x(pop):
    return [pop[i].x for i in range(len(pop))]


def get_f(pop):
    return [pop[i].f for i in range(len(pop))]


def print_pop(pop, rank, crowding, sorted_idx):
    for i in range(len(pop)):
        print i, pop[i].f, rank[sorted_idx[i]], crowding[sorted_idx[i]]
    print '---------'


