import time
from sklearn.cluster import AgglomerativeClustering

from pymoo.algorithms.experiment_online_cluster_nsga3 import ExperimentOnlineClusterNSGA3
from pymoo.algorithms.experiment_nsga3 import ExperimentNSGA3
from pymoo.algorithms.adapted_nsga3 import NSGA3
from pymoo.algorithms.online_cluster_nsga3 import OnlineClusterNSGA3
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# create the reference directions to be used for the optimization
original_dimension = 5
reduced_dimension = 2
interval_of_aggregations = 1
save_data = True
termination_criterion = ('n_gen', 20)
problem = get_problem("dtlz2", n_obj=original_dimension)
number_of_executions = 3
reference_directions = get_reference_directions("das-dennis", original_dimension, n_partitions=12)

start = time.time()

experiment = ExperimentOnlineClusterNSGA3(ref_dirs=reference_directions,
    pop_size=100,
    cluster=AgglomerativeClustering,
    save_dir='.\\experiment_results\\OnlineClusterNSGA3_{}_{}_{}_{}'.format(problem.name(), original_dimension, reduced_dimension, interval_of_aggregations),
    save_data=True,
    number_of_clusters=reduced_dimension,
    interval_of_aggregations=interval_of_aggregations,
    problem=problem,
    number_of_executions=number_of_executions,
    termination=termination_criterion,
    verbose=False,
    save_history=True,
    use_different_seeds=True)

print('Online Cluster NSGA-III Experiment Run')
experiment.run()
# experiment.show_mean_convergence('hv_convergence.txt')
experiment.show_mean_convergence('igd_convergence.txt')
experiment.show_heat_map()


experiment = ExperimentNSGA3(ref_dirs=reference_directions,
    pop_size=100,
    cluster=AgglomerativeClustering,
    save_dir='.\\experiment_results\\NSGA3_{}_{}'.format(problem.name(), original_dimension),
    save_data=True,
    number_of_clusters=reduced_dimension,
    interval_of_aggregations=interval_of_aggregations,
    problem=problem,
    number_of_executions=number_of_executions,
    termination=termination_criterion,
    verbose=False,
    save_history=True,
    use_different_seeds=True)

print('NSGA-III Experiment Run')
experiment.run()
# experiment.show_mean_convergence('hv_convergence.txt')
experiment.show_mean_convergence('igd_convergence.txt')

end = time.time()
print('Elapsed Time in experiment', end - start)