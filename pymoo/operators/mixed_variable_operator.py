import numpy as np

from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.sampling import Sampling


def prepare_processing(mask, operators):
    process = []

    # create a numpy array of mask if it is not yet
    mask = np.array(mask)

    for val in np.unique(mask):

        # check if operator for that type was defined
        if val not in operators:
            raise Exception("Operator for type %s was not defined." % val)

        # append it as a processing type
        process.append({
            "type": val,
            "mask": mask == val,
            "operator": operators[val]
        })

    return process


def apply_mixed_variable_operation(problem, process, fun):

    # the result to be returned
    ret = []

    # save the original bounds of the problem
    _n_var, _xl, _xu = problem.n_var, problem.xl, problem.xu

    # iterate through all the different operators that should be applied
    for entry in process:
        # get the mask and the operator
        mask, operator = entry["mask"], entry["operator"]

        # copy the arrays of the problem and cast them to float
        problem.n_var, problem.xl, problem.xu = mask.sum(), _xl[mask], _xu[mask]

        # perform the crossover
        ret.append(fun(mask, operator))

    # reset the original bounds of the problem
    problem.n_var = _n_var
    problem.xl = _xl
    problem.xu = _xu

    return ret


def concatenate_mixed_variables(problem, process, ret):
    # find the minimum of returned individuals and make them equal among operators
    n_rows = min([len(e) for e in ret])
    ret = [e[:n_rows] for e in ret]

    # create the result array and set the values for each operator
    X = np.full((n_rows, problem.n_var), np.nan, dtype=np.object)

    for i in range(len(process)):
        mask, _X = process[i]["mask"], ret[i]
        X[:, mask] = _X

    return X


class MixedVariableCrossover(Crossover):

    def __init__(self, mask, operators):

        n_parents = np.unique(np.array([op.n_parents for op in operators.values()]))
        if len(n_parents) > 1:
            raise Exception("All crossovers need to have the same number of parents!")

        n_offsprings = np.unique(np.array([op.n_offsprings for op in operators.values()]))
        if len(n_offsprings) > 1:
            raise Exception("All crossovers need to have the same number of offsprings!")

        super().__init__(n_parents[0], n_offsprings[0])
        self.process = prepare_processing(mask, operators)

    def _do(self, problem, X, **kwargs):

        _, n_matings, n_var = X.shape

        def fun(mask, operator):
            return operator._do(problem, X[..., mask], **kwargs)

        ret = apply_mixed_variable_operation(problem, self.process, fun)

        # for the crossover the concatenation is different through the 3d arrays.
        X = np.full((self.n_offsprings, n_matings, n_var), np.nan, dtype=np.object)
        for i in range(len(self.process)):
            mask, _X = self.process[i]["mask"], ret[i]
            X[..., mask] = _X

        return X


class MixedVariableMutation(Mutation):

    def __init__(self, mask, operators):
        super().__init__()
        self.process = prepare_processing(mask, operators)

    def _do(self, problem, X, **kwargs):
        def fun(mask, operator):
            return operator._do(problem, X[:, mask], **kwargs)

        ret = apply_mixed_variable_operation(problem, self.process, fun)
        return concatenate_mixed_variables(problem, self.process, ret)


class MixedVariableSampling(Sampling):

    def __init__(self, mask, operators):
        super().__init__()
        self.process = prepare_processing(mask, operators)

    def _do(self, problem, n_samples, **kwargs):
        def fun(mask, operator):
            return operator._do(problem, n_samples, **kwargs)

        ret = apply_mixed_variable_operation(problem, self.process, fun)
        return concatenate_mixed_variables(problem, self.process, ret)
