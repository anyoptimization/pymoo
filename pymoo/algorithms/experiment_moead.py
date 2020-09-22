from pymoo.algorithms.online_cluster_moead import OnlineClusterMOEAD
from pymoo.algorithms.adapted_moead import MOEAD 
from pymoo.factory import get_problem, get_visualization, get_reference_directions
from pymoo.optimize import minimize

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans

class ExperimentMOEAD(object):

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
                 show_heat_map=True,
                 **kwargs):

        self.save_data = save_data
        self.problem = problem
        self.termination = termination
        self.save_dir = save_dir
        self.verbose = verbose
        self.save_history = save_history
        self.show_heat_map = show_heat_map

        if use_different_seeds:
            self.algorithms  = MOEAD(ref_dirs,
                                        n_neighbors=n_neighbors,
                                        decomposition=decomposition,
                                        prob_neighbor_mating=prob_neighbor_mating,
                                        seed=i,
                                        current_execution_number=i,
                                        save_dir=self.save_dir,
                                        save_data=self.save_data,
                                        cluster=cluster) for i in range(self.number_of_executions)]
        else:
            self.algorithms  = [MOEAD(
                                        ref_dirs,
                                        n_neighbors=n_neighbors,
                                        decomposition=decomposition,
                                        prob_neighbor_mating=prob_neighbor_mating,
                                        seed=1,
                                        current_execution_number=i,
                                        save_dir=self.save_dir,
                                        save_data=self.save_data,
                                        cluster=cluster) for i in range(self.number_of_executions)]
    def run(self):
        results = []
        print('run is ok!')
        self.current_execution = 1
        for algorithm in self.algorithms:
            results.append(minimize(
                self.problem,
                 algorithm,
                 termination=self.termination,
                 verbose=self.verbose,
                 save_history=self.save_history))

            # get_visualization("scatter").add(res.F).show()
            self.current_execution +=1