from pymoo.algorithms.moead import MOEAD 
from pymoo.factory import get_problem, get_visualization, get_reference_directions
from pymoo.optimize import minimize
from sklearn.cluster import AgglomerativeClustering
from pymoo.algorithms.experiment_moead import ExperimentMOEAD

original_dimension = 4
n_neighbors=15
decomposition_type = 'pbi'
prob_neighbor_mating = 0.3
save_data = True
termination_criterion = ('n_gen', 20)
save_dir = '.\\experiment_results\\MOEAD'
number_of_executions = 3
reference_directions = get_reference_directions("das-dennis", original_dimension, n_partitions=12)
problem = get_problem("dtlz2", n_obj=original_dimension)

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