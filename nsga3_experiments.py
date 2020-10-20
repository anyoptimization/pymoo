import time
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from pymoo.algorithms.experiment_nsga3 import ExperimentNSGA3
from pymoo.algorithms.adapted_nsga3 import NSGA3
from pymoo.algorithms.online_cluster_nsga3 import OnlineClusterNSGA3
from pymoo.algorithms.offline_cluster_moead import OfflineClusterMOEAD
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.experiment_online_cluster_nsga3 import ExperimentOnlineClusterNSGA3
from pymoo.algorithms.experiment_offline_cluster_nsga3 import ExperimentOfflineClusterNSGA3
from pymoo.model.population import Population

def generate_transformation_matrix(seed, problem, ref_dirs):
    algorithm = OfflineClusterMOEAD(ref_dirs,
                            transformation_matrix=[[1]*problem.n_obj for i in range(problem.n_obj)],
                            seed=seed,
                            pop_size=100,
                            save_dir='',
                            save_data=False)

    res = minimize(problem, algorithm, termination=('n_gen', 0))
    return res.pop.get('F')

def generate_max_min(problem, reference_directions, number_of_executions):
    print('Generating random solutions for normalization...')
    populations = np.concatenate([generate_transformation_matrix(1, problem, reference_directions) for i in range(number_of_executions)])
    return populations.min(axis=0), populations.max(axis=0)

original_dimension = 5
reduced_dimension = 4
interval_of_aggregations = 1
save_data = True
use_normalization=True
termination_criterion = ('n_gen', 10)
problem = get_problem("dtlz2", n_obj=original_dimension)
number_of_executions = 3
reference_directions = get_reference_directions("das-dennis", original_dimension, n_partitions=12)

normalization_point = generate_max_min(problem, reference_directions, number_of_executions)
print('Values for normalization', normalization_point)

start = time.time()

experiment = ExperimentOnlineClusterNSGA3(ref_dirs=reference_directions,
    min_max_values=normalization_point,
    use_normalization=use_normalization,
    pop_size=100,
    cluster=AgglomerativeClustering,
    save_dir='.\\experiment_results\\OnlineClusterNSGA3_{}_{}_{}_{}'.format(problem.name(), original_dimension, reduced_dimension, interval_of_aggregations),
    save_data=True,
    number_of_clusters=reduced_dimension,
    interval_of_aggregations=interval_of_aggregations,
    problem=problem,
    number_of_executions=number_of_executions,
    termination=termination_criterion,
    use_random_aggregation=False,    
    verbose=False,
    save_history=True,
    use_different_seeds=True)

print('Online Cluster NSGA-III Experiment Run')
# experiment.run()
# experiment.show_mean_convergence('igd_convergence.txt')
# experiment.show_heat_map()


experiment = ExperimentNSGA3(ref_dirs=reference_directions,
    min_max_values=normalization_point,
    use_normalization=use_normalization,
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
experiment.show_mean_convergence('igd_convergence.txt')

experiment = ExperimentOnlineClusterNSGA3(ref_dirs=reference_directions,
    min_max_values=normalization_point,
    use_normalization=use_normalization,
    pop_size=100,
    cluster=AgglomerativeClustering,
    save_dir='.\\experiment_results\\RandomClusterNSGA3_{}_{}_{}_{}'.format(problem.name(), original_dimension, reduced_dimension, interval_of_aggregations),
    save_data=True,
    number_of_clusters=reduced_dimension,
    interval_of_aggregations=interval_of_aggregations,
    problem=problem,
    number_of_executions=number_of_executions,
    termination=termination_criterion,
    use_random_aggregation=True,    
    verbose=False,
    save_history=True,
    use_different_seeds=True)

print('Random Cluster NSGA-III Experiment Run')
experiment.run()
experiment.show_mean_convergence('igd_convergence.txt')
experiment.show_heat_map()

experiment = ExperimentOfflineClusterNSGA3(ref_dirs=reference_directions,
    min_max_values=normalization_point,
    use_normalization=use_normalization,
    pop_size=100,
    cluster=AgglomerativeClustering,
    save_dir='.\\experiment_results\\OfflineClusterNSGA3_{}_{}_{}'.format(problem.name(), original_dimension, reduced_dimension),
    save_data=True,
    number_of_clusters=reduced_dimension,
    interval_of_aggregations=interval_of_aggregations,
    problem=problem,
    number_of_executions=number_of_executions,
    termination=termination_criterion,
    use_random_aggregation=False,    
    verbose=False,
    save_history=True,
    use_different_seeds=True)

print('Offline Cluster NSGA-III Experiment Run')
experiment.run()
experiment.show_mean_convergence('igd_convergence.txt')
experiment.show_heat_map()


end = time.time()
print('Elapsed Time in experiment', end - start)