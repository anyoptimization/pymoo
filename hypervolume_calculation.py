import pymoo.model.population as P
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from pymoo.factory import get_problem, get_visualization, get_reference_directions, get_performance_indicator
from pymoo.optimize import minimize
from sklearn.cluster import AgglomerativeClustering

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

def evaluate_files_in_folder(metric, path, max_value, min_value):
    files = [file_name for file_name in os.listdir(path) if file_name.startswith('variable')]
    return[get_metric_from_file(hv, os.path.join(path, file_name), max_value, min_value, problem) 
    for file_name in files]

def evaluate_results(metric, save_dir, max_value, min_value, number_of_executions):
    return np.array([evaluate_files_in_folder(metric, os.path.join(save_dir, 'Execution {}'.format(execution)),
         max_value, min_value) for execution in range(number_of_executions)]).mean(axis=0)

def get_metric_from_file_without_normalization(metric, file, problem):
    data = read_file_for_metric(file)
    print('Current file {}'.format(file))
    return metric.calc(np.array(data))

def save_metrics(metric_list, path, file_name):
    with open(os.path.join(path, file_name), 'w') as file:
        for metric in metric_list:
            file.write(str(metric) + '\n')

def evaluate_files_in_folder_without_normalization(metric, path):
    files = [file_name for file_name in os.listdir(path) if file_name.startswith('objectives')]
    hvs = [get_metric_from_file_without_normalization(hv, os.path.join(path, file_name), problem) for file_name in files]
    save_metrics(hvs, path, 'hv_convergence.txt')
    return hvs

def evaluate_results_without_normalization(metric, save_dir, number_of_executions):
    mean_values = np.array([evaluate_files_in_folder_without_normalization(metric, os.path.join(save_dir, 'Execution {}'.format(execution))) 
                            for execution in range(number_of_executions)]).mean(axis=0)
    save_metrics(mean_values, save_dir, 'mean_hv_convergence.txt')
    return mean_values

def show_mean_convergence(save_dir, file_name, metrics):
    convergence = metrics
    plt.figure()
    plt.xlabel('Generation', fontsize=20)
    plt.ylabel(file_name.split('_')[0], fontsize=20)
    plt.plot(convergence)
    plt.title('Convergence', fontsize=20)
    plt.savefig(os.path.join(save_dir, file_name))
    plt.show()

original_dimension = 5
reduced_dimension = 4
interval_of_aggregations = 1
number_of_executions = 3
problem = get_problem("dtlz2", n_obj=original_dimension)
save_dir = '.\\experiment_results\\NSGA3_{}_{}'.format(problem.name(), original_dimension)
save_dir = '.\\experiment_results\\OnlineClusterNSGA3_{}_{}_{}_{}'.format(problem.name(), original_dimension, reduced_dimension, interval_of_aggregations)

reference_directions = get_reference_directions("das-dennis", original_dimension, n_partitions=12)
hv = get_performance_indicator("hv", ref_point=np.array([1.2]*problem.n_obj))
start = time.time()
hvs_values = evaluate_results_without_normalization(hv, save_dir, number_of_executions)
show_mean_convergence(save_dir, 'hv_convergence.pdf', hvs_values)
end = time.time()
print(hvs_values)

print('Elapsed time {}'.format(end-start))