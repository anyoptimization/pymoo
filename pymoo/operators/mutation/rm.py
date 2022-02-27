import numpy as np

from pymoo.core.mutation import VariableWiseMutation


class ChoiceRandomMutation(VariableWiseMutation):

    def _do(self, problem, X, **kwargs):
        assert problem.vars is not None

        prob_var = self.get_prob_var(problem, size=len(X))

        for k in range(problem.n_var):
            var = problem.vars[k]
            mut = np.where(np.random.random(len(X)) < prob_var)[0]
            X[mut, k] = var.sample(len(mut))

        return X

