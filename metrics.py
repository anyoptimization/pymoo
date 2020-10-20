from pymoo.algorithms.moead import MOEAD 
from pymoo.factory import get_problem, get_visualization, get_reference_directions, get_performance_indicator
from pymoo.optimize import minimize
from sklearn.cluster import AgglomerativeClustering
from pymoo.algorithms.experiment_moead import ExperimentMOEAD
from pymoo.algorithms.experiment_online_cluster_moead import ExperimentOnlineClusterMOEAD
from pymoo.algorithms.experiment_offline_cluster_moead import ExperimentOfflineClusterMOEAD
import pymoo.model.population as P
import os
import time
import numpy as np
import matplotlib.pyplot as plt

def read_file(file_path):
    lines = []
    with open(file_path, 'r') as file:
        lines = [float(number)  for line in file.readlines() for number in line.replace('[','').replace(']\n','').split()]
    return lines

def read_file_for_metric(file_path):
    lines = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    corrected_lines = ''.join(lines).replace('[','').replace('\n','').split(']')
    corrected_lines = [[float(number) for number in element.strip().split()] for element in corrected_lines if len(element) > 0]
    
    return np.array(corrected_lines)

def get_data_from_folder(folder):
    files = [i for i in os.listdir(folder) if i.startswith('objectives')]
    values = []
    for file in files:
        values += read_file(os.path.join(folder, file))  
    return np.array(values).flatten().tolist()

def get_max_min(root_path):
    executions = [i for i in os.listdir(root_path) if i.startswith('Execution')]
    data = []
    for folder in executions:
        data += get_data_from_folder(os.path.join(root_path, folder))
    return (np.max(data), np.min(data))

def get_metric_from_file(metric, file, max_value, min_value, problem):
    data = read_file_for_metric(file)
    data = (data - min_value)/(max_value - min_value)
    population = P.pop_from_array_or_individual(data)
    for individual in population:
        individual.F = problem.evaluate(individual.get('X'))
    return metric.calc(population.get('F'))

original_dimension = 4
reduced_dimension = 2
interval_of_aggregations = 1
problem = get_problem("dtlz2", n_obj=original_dimension)
save_dir = '.\\experiment_results\\MOEAD_{}_{}'.format(problem.name(), original_dimension)
number_of_executions = 3
online_save_dir = '.\\experiment_results\\OnlineClusterMOEAD_{}_{}_{}_{}'\
.format(problem.name(), original_dimension, reduced_dimension, interval_of_aggregations)

offline_save_dir = '.\\experiment_results\\OfflineClusterMOEAD_{}_{}_{}'\
.format(problem.name(), original_dimension, reduced_dimension)

moead_save_dir = '.\\experiment_results\\MOEAD_{}_{}'\
.format(problem.name(), original_dimension)

max_value_online, min_value_online = get_max_min(online_save_dir)
max_value_offline, min_value_offline = get_max_min(offline_save_dir)
max_value_moead, min_value_moead = get_max_min(moead_save_dir)

max_value = np.max([max_value_online, max_value_offline, max_value_moead])
min_value = np.max([min_value_online, min_value_offline, min_value_moead])
reference_directions = get_reference_directions("das-dennis", original_dimension, n_partitions=12)
hv = get_performance_indicator("hv", ref_point=np.array([1.0]*problem.n_obj))

# igd_plus = get_performance_indicator("igd+", problem.pareto_front(ref_dirs=reference_directions))

def evaluate_files_in_folder(metric, path, max_value, min_value):
    files = [file_name for file_name in os.listdir(path) if file_name.startswith('variable')]
    return[ get_metric_from_file(hv, os.path.join(path, file_name), max_value, min_value, problem) 
    for file_name in files]


def evaluate_results(metric, save_dir, max_value, min_value, number_of_executions):
    return np.array([evaluate_files_in_folder(metric, os.path.join(save_dir, 'Execution {}'.format(execution)),
         max_value, min_value) for execution in range(number_of_executions)]).mean(axis=0)

moead = evaluate_results(hv, moead_save_dir, max_value, min_value, number_of_executions)
online = evaluate_results(hv, online_save_dir, max_value, min_value, number_of_executions)
offline = evaluate_results(hv, offline_save_dir, max_value, min_value, number_of_executions)

plt.plot(moead, label='MOEAD')
plt.plot(online, label='OnlineMOEAD')
plt.plot(offline, label='OfflineMOEAD')
plt.xlabel('Generation', fontsize=20)
plt.ylabel('', fontsize=20)
plt.legend()
#plt.savefig(os.path.join(save_files_path, file_name.split('.')[0] + '.pdf'))
plt.show()

print(online_save_dir.split('\\')[2])