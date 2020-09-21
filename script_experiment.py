from pymoo.algorithms.online_cluster_moead import OnlineClusterMOEAD
from pymoo.algorithms.moead import MOEAD 
from pymoo.factory import get_problem, get_visualization, get_reference_directions
from pymoo.optimize import minimize
from sklearn.cluster import AgglomerativeClustering
from pymoo.algorithms.experiment_online_cluster_moead import ExperimentOnlineClusterMOEAD

original_dimension = 4
reduced_dimention = 2
n_neighbors=15
decomposition_type = 'pbi'
prob_neighbor_mating = 0.3
interval_of_aggregations = 1
save_data = True
termination_criterion = ('n_gen', 5)
save_dir = '.\\experiment_results\\OnlineClusterMOEAD'
number_of_executions = 3
reference_directions = get_reference_directions("das-dennis", original_dimension, n_partitions=12)
problem = get_problem("dtlz2", n_obj=original_dimension)
show_heat_map = True

experiment = ExperimentOnlineClusterMOEAD(
    reference_directions,
    n_neighbors=n_neighbors,
    decomposition=decomposition_type,
    prob_neighbor_mating=prob_neighbor_mating,
    number_of_clusters=reduced_dimention,
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
    show_heat_map=show_heat_map
    )

experiment.run()