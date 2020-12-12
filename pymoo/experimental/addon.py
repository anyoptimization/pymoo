import json
from urllib.request import urlopen

import numpy as np
import requests

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.model.infill import InfillCriterion
from pymoo.model.population import Population
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


def numpy_to_json(x):
    s = len(x.shape)
    if s == 1:
        return x.tolist()
    elif s == 2:
        return [e.tolist() for e in x]


def get_url():
    return urlopen("http://pymoo.org/addon.txt").read().decode('utf-8').replace("\n", "")


# =========================================================================================================
# Remote Calls
# =========================================================================================================


class RemoteInfillCriterion(InfillCriterion):

    def __init__(self, endpoint, params={}, **kwargs):
        super().__init__(**kwargs)
        self.endpoint = endpoint
        self.url = get_url()
        self.params = params

    def do(self, problem, pop, n_offsprings, **kwargs):
        X, F, G = pop.get("X", "F", "G")
        xl, xu = problem.bounds()

        # defining a params dict for the parameters to be sent to the API
        DATA = {'X': numpy_to_json(X),
                'F': numpy_to_json(F),
                'xl': numpy_to_json(xl),
                'xu': numpy_to_json(xu),
                }

        if problem.has_constraints():
            DATA['G'] = numpy_to_json(G)

        DATA = {**DATA,
                'n_infills': n_offsprings,
                **self.params}

        # sending get request and saving the response as response object
        r = requests.post(url=f"{self.url}/{self.endpoint}", json=json.dumps(DATA))

        # extracting data in json format
        resp = r.json()

        if not resp["success"]:
            raise Exception(f"ERROR during remote call: {resp['error']}")

        X = np.array(resp["X"])

        return Population.new(X=X)


class SANSGA2(NSGA2):

    def __init__(self, **kwargs):
        super().__init__(mating=RemoteInfillCriterion("sansga2"), **kwargs)


problem = get_problem("zdt1", n_var=10)

algorithm = SANSGA2(n_offsprings=10)

res = minimize(problem,
               algorithm,
               ('n_gen', 20),
               seed=1,
               verbose=True)

Scatter().add(res.F).show()
