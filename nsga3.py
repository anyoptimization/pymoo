import time
from sklearn.cluster import AgglomerativeClustering

from pymoo.algorithms.experiment_online_cluster_nsga3 import ExperimentOnlineClusterNSGA3
from pymoo.algorithms.adapted_nsga3 import NSGA3
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# create the reference directions to be used for the optimization
original_dimension = 5
save_data = True
termination_criterion = ('n_gen', 100)
problem = get_problem("dtlz2", n_obj=original_dimension)
number_of_executions = 3
reference_directions = get_reference_directions("das-dennis", original_dimension, n_partitions=12)


# create the algorithm object
algorithm = NSGA3(ref_dirs=reference_directions,
    pop_size=92,
    cluster=AgglomerativeClustering,
    save_dir='.\\experiment_results\\NSGA3',
    save_data=True)

start = time.time()

# execute the optimization
res = minimize(get_problem("dtlz2", n_obj=original_dimension),
               algorithm,
               seed=0,
               verbose=False,
               termination=('n_gen', 100))



end = time.time()
print('Elapsed Time in experiment', end - start)