import re

from pymoo.problems import *
from pymoo.model.problem import at_least2d, Problem

STR_TO_PROBLEM = {
    'ackley': Ackley,
    'bnh': BNH,
    'carside': Carside,
    'ctp1': CTP1,
    'ctp2': CTP2,
    'ctp3': CTP3,
    'ctp4': CTP4,
    'ctp5': CTP5,
    'ctp6': CTP6,
    'ctp7': CTP7,
    'ctp8': CTP8,
    'dtlz1_-1': InvertedDTLZ1,
    'dtlz1': DTLZ1,
    'dtlz2': DTLZ2,
    'dtlz3': DTLZ3,
    'dtlz4': DTLZ4,
    'dtlz5': DTLZ5,
    'dtlz6': DTLZ6,
    'dtlz7': DTLZ7,
    'c1dtlz1': C1DTLZ1,
    'c1dtlz3': C1DTLZ3,
    'c2dtlz2': C2DTLZ2,
    'c3dtlz4': C3DTLZ4,
    'cantilevered_beam': CantileveredBeam,
    'griewank': Griewank,
    'knp': Knapsack,
    'kursawe': Kursawe,
    'osy': OSY,
    'pressure_vessel': PressureVessel,
    'rastrigin': Rastrigin,
    'rosenbrock': Rosenbrock,
    'schwefel': Schwefel,
    'sphere': Sphere,
    'tnk': TNK,
    'truss2d': Truss2D,
    'welded_beam': WeldedBeam,
    'zakharov': Zakharov,
    'zdt1': ZDT1,
    'zdt2': ZDT2,
    'zdt3': ZDT3,
    'zdt4': ZDT4,
    'zdt6': ZDT6,
    'g01': G1,
    'g02': G2,
    'g03': G3,
    'g04': G4,
    'g05': G5,
    'g06': G6,
    'g07': G7,
    'g08': G8,
    'g09': G9,
    'g10': G10,
}


def get_problem(name, *args, return_list=False, **kwargs):
    # the resulting test single
    ret = []
    for e in STR_TO_PROBLEM:

        if re.match("(?i)" + e + "$", name):
            problem = STR_TO_PROBLEM[e](*args, **kwargs)
            ret.append((e, problem))

    # no problem that matched was found
    if len(ret) == 0:
        raise Exception("No problem that matched '%s' has been found." % name)

    # the whole list is returned if desired
    if return_list:
        return ret
    else:
        # multiple test single match
        if len(ret) > 1:
            raise Exception("Multiple test single match '%s': %s. Consider return_list if on intention"
                            % (name, [e[0] for e in ret]))
        # the normal case, only test problem was found
        else:
            return ret[0][1]


def get_problem_from_func(func, xl=None, xu=None, n_var=None, func_args={}):
    if xl is None or xu is None:
        raise Exception("Please provide lower and upper bounds for the problem.")
    if isinstance(xl, (int, float)):
        xl = xl * anp.ones(n_var)
    if isinstance(xu, (int, float)):
        xu = xu * anp.ones(n_var)

    # determine through a test evaluation details about the problem
    n_var = xl.shape[0]
    n_obj = -1
    n_constr = 0

    out = {}
    func(xl[None, :], out, **func_args)
    at_least2d(out)

    n_obj = out["F"].shape[1]
    if out.get("G") is not None:
        n_constr = out["G"].shape[1]

    class MyProblem(Problem):
        def __init__(self):
            Problem.__init__(self)
            self.n_var = n_var
            self.n_constr = n_constr
            self.n_obj = n_obj
            self.func = self._evaluate
            self.xl = xl
            self.xu = xu

        def _evaluate(self, x, out, *args, **kwargs):
            func(x, out, *args, **kwargs)

    return MyProblem()
