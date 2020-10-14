import time
from sklearn.cluster import AgglomerativeClustering

from pymoo.algorithms.experiment_online_cluster_nsga3 import ExperimentOnlineClusterNSGA3
from pymoo.algorithms.online_cluster_nsga3 import OnlineClusterNSGA3
from pymoo.algorithms.moead import MOEAD
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# create the reference directions to be used for the optimization
original_dimension = 5
reduced_dimension = 2
interval_of_aggregations = 1
save_data = True
termination_criterion = ('n_gen', 100)
problem = get_problem("dtlz2", n_obj=original_dimension)
number_of_executions = 3
reference_directions = get_reference_directions("das-dennis", original_dimension, n_partitions=12)


# create the algorithm object
algorithm = OnlineClusterNSGA3(ref_dirs=reference_directions,
    pop_size=92,
    cluster=AgglomerativeClustering,
    number_of_clusters=reduced_dimension,
    save_dir='.\\experiment_results\\OnlineClusterNSGA3',
    save_data=True)

print(algorithm.generate_max_min(problem))

start = time.time()

# execute the optimization
# res = minimize(get_problem("dtlz2", n_obj=original_dimension),
#                algorithm,
#                seed=5,
#                verbose=False,
#                termination=('n_gen', 200))


experiment = ExperimentOnlineClusterNSGA3(ref_dirs=reference_directions,
    pop_size=92,
    cluster=AgglomerativeClustering,
    save_dir='.\\experiment_results\\OnlineClusterNSGA3',
    save_data=True,
    number_of_clusters=reduced_dimension,
    interval_of_aggregations=interval_of_aggregations,
    problem=problem,
    number_of_executions=number_of_executions,
    termination=termination_criterion,
    verbose=False,
    save_history=True,
    use_different_seeds=True)

print('Experiment run')
experiment.run()
experiment.show_mean_convergence('hv_convergence.txt')
experiment.show_mean_convergence('igd_convergence.txt')
experiment.show_heat_map()

end = time.time()
print('Elapsed Time in experiment', end - start)