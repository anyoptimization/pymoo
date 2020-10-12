import time
from sklearn.cluster import AgglomerativeClustering

from pymoo.algorithms.online_cluster_nsga3 import OnlineClusterNSGA3
from pymoo.algorithms.moead import MOEAD
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

number_of_objectives = 5

# create the reference directions to be used for the optimization
ref_dirs = get_reference_directions("das-dennis", number_of_objectives, n_partitions=12)

# create the algorithm object
algorithm = OnlineClusterNSGA3(ref_dirs=ref_dirs,
    pop_size=92,
    cluster=AgglomerativeClustering)

start = time.time()

# execute the optimization
res = minimize(get_problem("dtlz2", n_obj=number_of_objectives),
               algorithm,
               seed=1,
               verbose=False,
               termination=('n_gen', 20))

end = time.time()
print('Elapsed Time in experiment', end - start)

Scatter().add(res.F).show()