from pymoo.algorithms.online_cluster_moead import OnlineClusterMOEAD
from pymoo.algorithms.adapted_nsga3 import NSGA3
from pymoo.factory import get_problem, get_visualization, get_reference_directions
from pymoo.optimize import minimize

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans

class ExperimentNSGA3(object):

    def __init__(self,
                 ref_dirs,
                 n_neighbors=20,
                 decomposition='auto',
                 prob_neighbor_mating=0.9,
                 save_data=True,
                 problem=get_problem("dtlz1"),
                 number_of_executions=1,
                 termination=('n_gen',100),
                 save_dir='',
                 verbose=False,
                 save_history=True,
                 use_different_seeds=True,
                 **kwargs):

        self.save_data = save_data
        self.problem = problem
        self.termination = termination
        self.save_dir = save_dir
        self.verbose = verbose
        self.save_history = save_history
        self.number_of_executions = number_of_executions
        if use_different_seeds:
            self.algorithms  = [NSGA3(
                                        ref_dirs,
                                        pop_size=100,
                                        n_neighbors=n_neighbors,
                                        decomposition=decomposition,
                                        prob_neighbor_mating=prob_neighbor_mating,
                                        seed=i,
                                        current_execution_number=i,
                                        save_dir=self.save_dir,
                                        save_data=self.save_data) for i in range(self.number_of_executions)]
        else:
            self.algorithms  = [NSGA3(
                                        ref_dirs,
                                        pop_size=100,
                                        n_neighbors=n_neighbors,
                                        decomposition=decomposition,
                                        prob_neighbor_mating=prob_neighbor_mating,
                                        seed=1,
                                        current_execution_number=i,
                                        save_dir=self.save_dir,
                                        save_data=self.save_data) for i in range(self.number_of_executions)]
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