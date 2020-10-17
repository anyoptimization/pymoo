from pymoo.algorithms.online_cluster_nsga3 import OnlineClusterNSGA3
from pymoo.factory import get_problem, get_visualization, get_reference_directions
from pymoo.optimize import minimize

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans

class ExperimentOnlineClusterNSGA3(object):

    def __init__(self,
                 ref_dirs,
                 n_neighbors=20,
                 decomposition='auto',
                 prob_neighbor_mating=0.9,
                 cluster=KMeans,
                 number_of_clusters=2,
                 interval_of_aggregations=1,
                 save_data=True,
                 problem=get_problem("dtlz1"),
                 number_of_executions=1,
                 termination=('n_gen', 100),
                 use_random_aggregation=False,
                 save_dir='',
                 verbose=False,
                 save_history=True,
                 use_different_seeds=True,
                 **kwargs):

        self.save_data = save_data
        self.problem = problem
        self.number_of_executions = number_of_executions
        self.termination = termination
        self.use_random_aggregation = use_random_aggregation
        self.save_dir = save_dir
        self.verbose = verbose
        self.save_history = save_history

        if use_different_seeds:
            self.algorithms  = [OnlineClusterNSGA3(
                                        ref_dirs,
                                        pop_size=100,
                                        n_neighbors=n_neighbors,
                                        decomposition=decomposition,
                                        prob_neighbor_mating=prob_neighbor_mating,
                                        seed=i,
                                        number_of_clusters=number_of_clusters,
                                        number_of_clusters_for_directions=number_of_clusters,
                                        interval_of_aggregations=interval_of_aggregations,
                                        current_execution_number=i,
                                        use_random_aggregation = use_random_aggregation,
                                        save_dir=self.save_dir,
                                        save_data=self.save_data,
                                        cluster=cluster) for i in range(self.number_of_executions)]
        else:
            self.algorithms  = [OnlineClusterNSGA3(
                                        ref_dirs,
                                        pop_size=100,
                                        n_neighbors=n_neighbors,
                                        decomposition=decomposition,
                                        prob_neighbor_mating=prob_neighbor_mating,
                                        seed=1,
                                        number_of_clusters=number_of_clusters,
                                        number_of_clusters_for_directions=number_of_clusters,
                                        interval_of_aggregations=interval_of_aggregations,
                                        current_execution_number=i,
                                        use_random_aggregation = use_random_aggregation,
                                        save_dir=self.save_dir,
                                        save_data=self.save_data,
                                        cluster=cluster) for i in range(self.number_of_executions)]
    def run(self):
        results = []
        self.current_execution = 1
        for algorithm in self.algorithms:
            print('CURRENT EXECUTION {}'.format(self.current_execution))
            results.append(minimize(
                self.problem,
                 algorithm,
                 termination=self.termination,
                 verbose=self.verbose,
                 save_history=self.save_history))

            self.current_execution +=1

    def show_heat_map(self):
        aggregations = []
        for i in range(self.number_of_executions):
            aggregations.append(pd.read_csv(os.path.join(self.save_dir,'Execution {}'.format(i), 'aggregations.txt'), header=None))

        aggregations = pd.concat(aggregations, axis=1)
        aggregations.columns = ['exec_{}'.format(i) for i in range(self.number_of_executions)]
        aggregation_list = [aggregations['exec_{}'.format(i)].value_counts().keys().values.tolist() 
                            for i in range(self.number_of_executions)]
        unique_aggregations = list(set([j  for i in aggregation_list for j in i]))
        unique_aggregations.sort(key = lambda x: x.split('-')[1], reverse=True)
        data_transposed = aggregations.T

        number_of_aggregations = len(unique_aggregations)
        number_of_generations = len(aggregations.index)
        heat_map = pd.DataFrame(data=np.zeros((number_of_aggregations, number_of_generations)))
        heat_map.index = unique_aggregations

        for i in range(number_of_generations):
            for k,v in data_transposed[i].value_counts().items():
                heat_map.at[k, i] = v

        for i in range(number_of_generations):
            if heat_map[i].values.sum() != self.number_of_executions:
                print('Error in generation {}'.format(i))

        plt.figure(figsize=(18,10))
        sns.heatmap(heat_map.values, yticklabels=heat_map.index.values, cmap="Blues")

        plt.xlabel('Generation', fontsize=20)

        plt.yticks(fontsize=13)
        plt.ylabel('Aggregation', fontsize=20)

        plt.title('Aggregation Heat Map', fontsize=20)
        plt.savefig(os.path.join(self.save_dir, 'heat_map.pdf'))
        plt.show()

    def show_mean_convergence(self, file_name):
        convergence = pd.DataFrame(self.generate_mean_convergence(file_name)).mean().values
        self.save_convergence(file_name, convergence)
        plt.figure()
        plt.xlabel('Generation', fontsize=20)
        plt.ylabel(file_name.split('_')[0], fontsize=20)
        plt.plot(convergence)
        plt.title('Convergence', fontsize=20)
        plt.savefig(os.path.join(self.save_dir, file_name.split('.')[0] + '.pdf'))
        plt.show()

    def generate_mean_convergence(self, file_name):
        return [self.read_data_file(os.path.join(self.save_dir, 'Execution {}'.format(execution), file_name)) 
                                for execution in range(self.number_of_executions)]      
            
    def read_data_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = [float(line.replace('\n','')) for line in file.readlines()]
            return lines

    def save_convergence(self, file_name, convergence):
        with open(os.path.join(self.save_dir, 'mean_' + file_name), 'w') as file:
            file.write('\n'.join([str(i) for i in convergence]))