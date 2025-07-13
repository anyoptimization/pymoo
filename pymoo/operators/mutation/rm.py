import numpy as np

from pymoo.core.mutation import Mutation


class ChoiceRandomMutation(Mutation):

    def _do(self, problem, X, random_state=None, **kwargs):
        assert problem.vars is not None

        # ensure the type object (fixed string length <UX can cause issues)
        X = X.astype(object)

        prob_var = self.get_prob_var(problem, size=len(X))

        for k, (_, var) in enumerate(problem.vars.items()):
            mut = np.where(random_state.random(len(X)) < prob_var)[0]

            v = var.sample(len(mut), random_state=random_state)
            X[mut, k] = v

        return X

