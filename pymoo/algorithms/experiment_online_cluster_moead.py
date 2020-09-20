from pymoo.algorithms.online_cluster_moead import OnlineClusterMOEAD
from pymoo.algorithms.moead import MOEAD 
from pymoo.factory import get_problem, get_visualization, get_reference_directions
from pymoo.optimize import minimize
from sklearn.cluster import AgglomerativeClustering, KMeans

class ExperimentOnlineClusterMOEAD(object):

    def __init__(self,
                 ref_dirs,
                 n_neighbors=20,
                 decomposition='auto',
                 prob_neighbor_mating=0.9,
                 cluster=KMeans,
                 number_of_clusters=2,
                 interval_of_aggregations=1,
                 save_algorith_current_iteration_data=True,
                 problem=get_problem("dtlz1"),
                 number_of_executions=1,
                 termination=('n_gen',100),
                 verbose=False,
                 save_history=True,
                 use_different_seeds=True,
                 show_heat_map=True,
                 **kwargs):

        self.save_algorith_current_iteration_data = save_algorith_current_iteration_data
        self.problem = problem
        self.number_of_executions = number_of_executions
        self.termination = termination
        self.verbose = verbose
        self.save_history = save_history
        self.show_heat_map = show_heat_map
        
        if use_different_seeds:
            self.algorithms  = [OnlineClusterMOEAD(
                                        ref_dirs,
                                        n_neighbors=n_neighbors,
                                        decomposition=decomposition,
                                        prob_neighbor_mating=prob_neighbor_mating,
                                        seed=i,
                                        number_of_clusters=number_of_clusters,
                                        interval_of_aggregations=interval_of_aggregations,
                                        save_current_iteration_data=self.save_algorith_current_iteration_data,
                                        cluster=cluster) for i in range(self.number_of_executions)]
        else:
            self.algorithms  = [OnlineClusterMOEAD(
                                        ref_dirs,
                                        n_neighbors=n_neighbors,
                                        decomposition=decomposition,
                                        prob_neighbor_mating=prob_neighbor_mating,
                                        seed=1,
                                        number_of_clusters=number_of_clusters,
                                        interval_of_aggregations=interval_of_aggregations,
                                        save_current_iteration_data=self.save_algorith_current_iteration_data,
                                        cluster=cluster) for i in range(self.number_of_executions)]
    def run(self):
        self.current_execution = 1
        for algorithm in self.algorithms:
            res = minimize(
                self.problem,
                 algorithm,
                 termination=self.termination,
                 verbose=self.verbose,
                 save_history=self.save_history)

            get_visualization("scatter").add(res.F).show()
            self.save_current_execution_files()
            self.current_execution +=1

        #generate heatmap if number of executions is greater than 1
        if self.show_heat_map:
            self.generate_aggregation_heat_map()
    
    def save_current_execution_files(self):
        
        pass

    def generate_aggregation_heat_map(self):
        pass