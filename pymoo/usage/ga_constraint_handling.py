from pymoo.optimize import minimize
from pymop.factory import get_problem

problem = get_problem("rastrigin", n_var=10)

res = minimize(problem,
               method='de',
               method_args={
                   'pop_size': 200,
                   'variant': "DE/rand+best/1/exp",
                   'CR': 0.5,
                   'F': 0.75,
                   # 'selection': RandomSelection(),
                   # 'survival': ConstraintHandlingSurvival(method="parameter_less"),
                   # 'survival': ConstraintHandlingSurvival(method="epsilon_constrained")
                   # 'survival': ConstraintHandlingSurvival(method="penalty", weight=0.25)
                   # 'survival': ConstraintHandlingSurvival(method="stochastic_ranking", prob=0.45)
               },
               termination=('n_gen', 1750),
               disp=True)

print("Best solution found: %s" % res.X)
print("Function value: %s" % res.F)
