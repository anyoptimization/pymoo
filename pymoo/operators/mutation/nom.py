from pymoo.model.mutation import Mutation


class NoMutation(Mutation):

    def _do(self, problem, X, **kwargs):
        return X
