import time
import numpy as np
import pandas as pd
from pymoo.optimize import minimize
from sklearn.cluster import AgglomerativeClustering
from pymoo.factory import get_problem, get_reference_directions
from pymoo.algorithms.offline_cluster_moead import OfflineClusterMOEAD

original_dimension = 4
reduced_dimension = 2
interval_of_aggregations = 1
n_neighbors=10
decomposition_type = 'pbi'
prob_neighbor_mating = 0.002#0.3
save_data = True
termination_criterion = ('n_gen', 0)
problem = get_problem("dtlz2", n_obj=original_dimension)
save_dir = '.\\experiment_results\\TEST_PF\\MOEAD_{}_{}'.format(problem.name(), original_dimension)
reference_directions = get_reference_directions("das-dennis", original_dimension, n_partitions=12)


algorithm = OfflineClusterMOEAD(reference_directions,
                                n_neighbors=n_neighbors,
                                decomposition=decomposition_type,
                                prob_neighbor_mating=prob_neighbor_mating,
                                seed=1,
                                pop_size=10000,
                                number_of_clusters=reduced_dimension,
                                current_execution_number=0,
                                save_dir=save_dir,
                                save_data=save_data,
                                cluster=AgglomerativeClustering)

res = minimize(problem, algorithm, termination=termination_criterion)
dataframe = pd.DataFrame(np.array(res.pop.get('F')))
similarity = 1 - dataframe.corr(method='kendall').values
cluster = AgglomerativeClustering(n_clusters=reduced_dimension, affinity='precomputed', linkage='single')
cluster.fit(similarity)
print(pd.get_dummies(cluster.labels_).T.values)
