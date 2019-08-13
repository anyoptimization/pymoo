from pymoo.factory import get_problem

p = get_problem("dtlz1_-1", n_var=20, n_obj=5)

# create a simple test problem from string
p = get_problem("Ackley")

# the input name is not case sensitive
p = get_problem("ackley")

