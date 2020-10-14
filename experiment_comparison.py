import os
import pandas as pd
import matplotlib.pyplot as plt

def read_data_file(file_path):
    with open(file_path, 'r') as file:
        lines = [float(line.replace('\n','')) for line in file.readlines()]
        return lines

def generate_convergence_plot(file_name, save_files_path, results_path_1, algorithm_1,
 results_path_2, algorithm_2):
    metric_1 = read_data_file(os.path.join(results_path_1, algorithm_1, file_name))
    metric_2 = read_data_file(os.path.join(results_path_2, algorithm_2, file_name))
    plt.plot(metric_1, label=algorithm_1)
    plt.plot(metric_2, label=algorithm_2)
    plt.xlabel('Generation', fontsize=20)
    plt.ylabel(file_name.split('.')[0], fontsize=20)
    plt.legend()
    plt.savefig(os.path.join(save_files_path, file_name.split('.')[0] + '.pdf'))
    plt.show()

def generate_convergence_plot_full(file_name, save_files_path, results_path_1, algorithm_1,
 results_path_2, algorithm_2, results_path_3, algorithm_3):
    metric_1 = read_data_file(os.path.join(results_path_1, algorithm_1, file_name))
    metric_2 = read_data_file(os.path.join(results_path_2, algorithm_2, file_name))
    metric_3 = read_data_file(os.path.join(results_path_3, algorithm_3, file_name))
    plt.plot(metric_1, label=algorithm_1)
    plt.plot(metric_2, label=algorithm_2)
    plt.plot(metric_3, label=algorithm_3)
    plt.xlabel('Generation', fontsize=20)
    plt.ylabel(file_name.split('.')[0], fontsize=20)
    plt.legend()
    plt.savefig(os.path.join(save_files_path, file_name.split('.')[0] + '.pdf'))
    plt.show()

def generate_boxplots(number_of_executions, file_name, save_files_path, results_path_1, algorithm_1,
 results_path_2, algorithm_2):
    data_1 = []
    data_2 = []
    for i in range(number_of_executions):
        data_1.append(read_data_file(
            os.path.join(results_path_1, algorithm_1, 'Execution {}'.format(i), file_name))[-1])
        data_2.append(read_data_file(
            os.path.join(results_path_2, algorithm_2, 'Execution {}'.format(i), file_name))[-1])
    
    plt.title(file_name.split('_')[0])
    plt.boxplot([data_1, data_2], labels=[algorithm_1, algorithm_2])
    plt.savefig(os.path.join(save_files_path, file_name.split('_')[0] + '_boxplot.pdf'))
    plt.show()

def generate_boxplots_full(number_of_executions, file_name, save_files_path, results_path_1, algorithm_1,
 results_path_2, algorithm_2, results_path_3, algorithm_3):
    data_1 = []
    data_2 = []
    data_3 = []
    for i in range(number_of_executions):
        data_1.append(read_data_file(
            os.path.join(results_path_1, algorithm_1, 'Execution {}'.format(i), file_name))[-1])
        data_2.append(read_data_file(
            os.path.join(results_path_2, algorithm_2, 'Execution {}'.format(i), file_name))[-1])
        data_3.append(read_data_file(
            os.path.join(results_path_3, algorithm_3, 'Execution {}'.format(i), file_name))[-1])
    
    plt.title(file_name.split('_')[0])
    plt.boxplot([data_1, data_2, data_3], labels=[algorithm_1, algorithm_2, algorithm_3])
    plt.savefig(os.path.join(save_files_path, file_name.split('_')[0] + '_boxplot.pdf'))
    plt.show()

def create_folder(full_path):
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print('Final results folder created!')
    else:
        print('Folder already exists!')

problem = 'DTLZ2'
original_dimension = 4
reduced_dimension = 3
number_of_executions = 3
interval_of_aggregations = 1
algorithm_1 = 'MOEAD_{}_{}'.format(problem, original_dimension)
results_path_1 = '.\\experiment_results\\TEST'
algorithm_2 = 'OnlineClusterMOEAD_{}_{}_{}_{}'.format(problem, original_dimension, reduced_dimension, interval_of_aggregations)
results_path_2 = '.\\experiment_results\\TEST'
algorithm_3 = 'OfflineClusterMOEAD_{}_{}_{}'.format(problem, original_dimension, reduced_dimension)
results_path_3 = '.\\experiment_results\\TEST'
save_files_path = '.\\experiment_results\\{}_{}_{}_{}'.format(problem, original_dimension, reduced_dimension, interval_of_aggregations)
hv_file_name = 'mean_hv_convergence.txt'
igd_file_name = 'mean_igd_convergence.txt'

create_folder(save_files_path)

generate_convergence_plot_full(hv_file_name, save_files_path, results_path_1, algorithm_1,
 results_path_2, algorithm_2, results_path_3, algorithm_3)
generate_convergence_plot_full(igd_file_name, save_files_path, results_path_1, algorithm_1,
 results_path_2, algorithm_2, results_path_3, algorithm_3)

generate_boxplots_full(number_of_executions, 'hv_convergence.txt', save_files_path, results_path_1, algorithm_1,
 results_path_2, algorithm_2, results_path_3, algorithm_3)
generate_boxplots_full(number_of_executions, 'igd_convergence.txt',save_files_path, results_path_1, algorithm_1,
 results_path_2, algorithm_2, results_path_3, algorithm_3)