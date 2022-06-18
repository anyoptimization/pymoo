from pymoo.core.mutation import Mutation


class NoMutation(Mutation):

    def do(self, problem, pop, **kwargs):
        return pop
