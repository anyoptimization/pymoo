from pymoo.algorithms.moead import MOEAD 
from pymoo.factory import get_problem, get_visualization, get_reference_directions
from pymoo.optimize import minimize
from sklearn.cluster import AgglomerativeClustering
from pymoo.algorithms.experiment_moead import ExperimentMOEAD
from pymoo.algorithms.experiment_online_cluster_moead import ExperimentOnlineClusterMOEAD

import time

original_dimension = 4
reduced_dimension = 2
interval_of_aggregations = 1
n_neighbors=10#15
decomposition_type = 'pbi'
prob_neighbor_mating = 0.002#0.3
save_data = True
termination_criterion = ('n_gen', 400)
problem = get_problem("dtlz2", n_obj=original_dimension)
save_dir = '.\\experiment_results\\MOEAD_{}_{}'.format(problem.name(), original_dimension)
number_of_executions = 11
reference_directions = get_reference_directions("das-dennis", original_dimension, n_partitions=12)

start = time.time()
experiment = ExperimentMOEAD(
    reference_directions,
    n_neighbors=n_neighbors,
    decomposition=decomposition_type,
    prob_neighbor_mating=prob_neighbor_mating,
    problem=problem,
    number_of_executions=number_of_executions,
    termination=termination_criterion,
    save_dir=save_dir,
    save_data=save_data,
    verbose=False,
    save_history=True,
    use_different_seeds=True,
    )

experiment.run()
experiment.show_mean_convergence('hv_convergence.txt')
experiment.show_mean_convergence('igd_convergence.txt')

# online cluster moead parameters
save_data = True
save_dir = '.\\experiment_results\\OnlineClusterMOEAD_{}_{}_{}'.format(problem.name(), original_dimension, reduced_dimension)
show_heat_map = True

experiment = ExperimentOnlineClusterMOEAD(
    reference_directions,
    n_neighbors=n_neighbors,
    decomposition=decomposition_type,
    prob_neighbor_mating=prob_neighbor_mating,
    number_of_clusters=reduced_dimension,
    interval_of_aggregations=interval_of_aggregations,
    problem=problem,
    cluster=AgglomerativeClustering,
    number_of_executions=number_of_executions,
    termination=termination_criterion,
    save_dir=save_dir,
    save_data=save_data,
    verbose=False,
    save_history=True,
    use_different_seeds=True,
    )

experiment.run()
experiment.show_mean_convergence('hv_convergence.txt')
experiment.show_mean_convergence('igd_convergence.txt')
experiment.show_heat_map()

end = time.time()
print('Elapsed Time in experiment', end - start)