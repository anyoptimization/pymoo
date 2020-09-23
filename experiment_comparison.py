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
    
problem = 'DTLZ2'
original_dimension = 4
reduced_dimension = 2
algorithm_1 = 'MOEAD'
results_path_1 = '.\\experiment_results'
algorithm_2 = 'OnlineClusterMOEAD'
results_path_2 = '.\\experiment_results'
save_files_path = '.\\experiment_results'
hv_file_name = 'mean_hv_convergence.txt'
igd_file_name = 'mean_igd_convergence.txt'

generate_convergence_plot(hv_file_name, save_files_path, results_path_1, algorithm_1, results_path_2, algorithm_2)
generate_convergence_plot(igd_file_name, save_files_path, results_path_1, algorithm_1, results_path_2, algorithm_2)

generate_boxplots(3, 'hv_convergence.txt', save_files_path, results_path_1, algorithm_1,
 results_path_2, algorithm_2)
generate_boxplots(3, 'igd_convergence.txt',save_files_path, results_path_1, algorithm_1,
 results_path_2, algorithm_2)