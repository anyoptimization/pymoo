from pymoo.algorithms.offline_cluster_moead import OfflineClusterMOEAD
from pymoo.algorithms.moead import MOEAD 
from pymoo.factory import get_problem, get_visualization, get_reference_directions
from pymoo.optimize import minimize

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
import time

class ExperimentOfflineClusterMOEAD(object):

    def __init__(self,
                 ref_dirs,
                 n_neighbors=20,
                 decomposition='auto',
                 prob_neighbor_mating=0.9,
                 cluster=KMeans,
                 number_of_clusters=2,
                 save_data=True,
                 problem=get_problem("dtlz1"),
                 number_of_executions=1,
                 termination=('n_gen', 100),
                 save_dir='',
                 verbose=False,
                 save_history=True,
                 use_different_seeds=True,
                 **kwargs):

        self.save_data = save_data
        self.problem = problem
        self.number_of_executions = number_of_executions
        self.termination = termination
        self.save_dir = save_dir
        self.verbose = verbose
        self.save_history = save_history

        self.ref_dirs = ref_dirs
        self.n_neighbors = n_neighbors
        self.prob_neighbor_mating = prob_neighbor_mating
        self.decomposition = decomposition
        self.cluster = cluster
        self.number_of_clusters = number_of_clusters
        
        self.transformation_matrix = self.generate_transformation_matrix()
        if use_different_seeds:
            self.algorithms  = [OfflineClusterMOEAD(
                                        ref_dirs,
                                        n_neighbors=n_neighbors,
                                        decomposition=decomposition,
                                        prob_neighbor_mating=prob_neighbor_mating,
                                        seed=i,
                                        number_of_clusters=number_of_clusters,
                                        transformation_matrix=self.transformation_matrix,
                                        current_execution_number=i,
                                        save_dir=self.save_dir,
                                        save_data=self.save_data,
                                        cluster=cluster) for i in range(self.number_of_executions)]
        else:
            self.algorithms  = [OfflineClusterMOEAD(
                                        ref_dirs,
                                        n_neighbors=n_neighbors,
                                        decomposition=decomposition,
                                        prob_neighbor_mating=prob_neighbor_mating,
                                        seed=1,
                                        number_of_clusters=number_of_clusters,
                                        transformation_matrix=self.transformation_matrix,
                                        current_execution_number=i,
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

    def generate_transformation_matrix(self):
        print('Generating random solutions for aggregations...')
        algorithm = OfflineClusterMOEAD(self.ref_dirs,
                                transformation_matrix=[[1]*self.problem.n_obj for i in range(self.problem.n_obj)],
                                n_neighbors=self.n_neighbors,
                                decomposition=self.decomposition,
                                prob_neighbor_mating=self.prob_neighbor_mating,
                                seed=1,
                                pop_size=10000,
                                number_of_clusters=self.number_of_clusters,
                                current_execution_number=0,
                                save_dir=self.save_dir,
                                save_data=self.save_data,
                                cluster=self.cluster)

        res = minimize(self.problem, algorithm, termination=('n_gen', 0))
        print('Random solutions generated...')
        print('Start clustering...')
        dataframe = pd.DataFrame(np.array(res.pop.get('F')))
        similarity = 1 - dataframe.corr(method='kendall').values
        cluster = self.cluster(n_clusters=self.number_of_clusters, affinity='precomputed', linkage='single')
        cluster.fit(similarity)
        print('Cluster generated...')
        transformation_matrix = pd.get_dummies(cluster.labels_).T.values
        print('Transformation matrix:')
        print(transformation_matrix)
        return transformation_matrix

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
        #plt.show()

    def show_mean_convergence(self, file_name):
        convergence = pd.DataFrame(self.generate_mean_convergence(file_name)).mean().values
        self.save_convergence(file_name, convergence)
        plt.figure()
        plt.xlabel('Generation', fontsize=20)
        plt.ylabel(file_name.split('_')[0], fontsize=20)
        plt.plot(convergence)
        plt.title('Convergence', fontsize=20)
        plt.savefig(os.path.join(self.save_dir, file_name.split('.')[0] + '.pdf'))
        #plt.show()

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