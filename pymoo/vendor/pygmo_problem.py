import pygmo as pg

def create_pygmo_problem(my_problem):
    func = PygmoFunction(my_problem)
    return pg.problem(func)


class PygmoFunction:

    def __init__(self, my_problem):
        self.my_problem = my_problem

    def fitness(self, x):
        f,g =  self.my_problem.evaluate(x)
        return f

    def get_bounds(self):
        return (self.my_problem.xl, self.my_problem.xu)

    def get_nobj(self):
        return self.my_problem.n_obj