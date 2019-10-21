from pymoo.factory import get_problem

from pymoo.vendor.global_opt import get_global_optimization_problem_options

problems = get_global_optimization_problem_options()

for e in problems:
    name = e[2]["clazz"].__name__
    string = e[0]

    p = get_problem(e[0])

    n_constr = p.n_constr
    if n_constr == 0:
        n_constr = ""

    print('|%s|%s|%s|"%s"|' % (name, p.n_var, n_constr, string))
