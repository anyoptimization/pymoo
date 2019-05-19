from pymoo.factory import get_problem

p = get_problem("dtlz1_-1", n_var=20, n_obj=5)

# create a simple test problem from string
p = get_problem("Ackley")

# the input name is not case sensitive
p = get_problem("ackley")

# also input parameter can be provided directly
p = get_problem("dtlz1_-1", n_var=20, n_obj=5)

# in case several test single should be loaded with a prefix this is possible as well
#p = get_problem("dtlz", return_list=True)

