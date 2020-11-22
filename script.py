from pymoo.algorithms.online_cluster_moead import OnlineClusterMOEAD
from pymoo.algorithms.moead import MOEAD 
from pymoo.factory import get_problem, get_visualization, get_reference_directions
from pymoo.optimize import minimize
from sklearn.cluster import AgglomerativeClustering

problem = get_problem("dtlz2", n_obj=4)

# remember to calculate metrics over non-dominated set, not in all current population
algorithm = OnlineClusterMOEAD(
    get_reference_directions("das-dennis", 4, n_partitions=12),
    n_neighbors=15,
    decomposition="pbi",
    prob_neighbor_mating=0.3,
    seed=1,
    number_of_clusters=2,
    interval_of_aggregations=10,
    cluster=AgglomerativeClustering
)
res = minimize(problem, algorithm, termination=('n_gen', 10), verbose=False, save_history=True)
print(res.pf)
print(res.opt)
print(res.pop)
print('Lengths', len(res.opt), len(res.pop))
#get_visualization("scatter").add(res.F).show()
