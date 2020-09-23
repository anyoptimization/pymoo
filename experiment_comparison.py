import os
import pandas as pd
import matplotlib.pyplot as plt

#plot convergence for hv
#plot convergence for igd
#plot boxplot for hv
#plot boxplot for igd

def read_data_file(file_path):
    with open(file_path, 'r') as file:
        lines = [float(line.replace('\n','')) for line in file.readlines()]
        return lines

def generate_convergence_plot(file_name, results_path_1, algorithm_1, results_path_2, algorithm_2):
    metric_1 = read_data_file(os.path.join(results_path_1, algorithm_1, file_name))
    metric_2 = read_data_file(os.path.join(results_path_2, algorithm_2, file_name))
    plt.plot(metric_1, label=algorithm_1)
    plt.plot(metric_2, label=algorithm_2)
    plt.xlabel('Generation', fontsize=20)
    plt.ylabel(file_name.split('.')[0], fontsize=20)
    plt.legend()
    plt.savefig(os.path.join(save_files_path, file_name.split('.')[0] + '.pdf'))
    plt.show()

def generate_boxplots(results_path_1, algorithm_1, results_path_2, algorithm_2):
    pass

algorithm_1 = 'MOEAD'
results_path_1 = '.\\experiment_results'
algorithm_2 = 'OnlineClusterMOEAD'
results_path_2 = '.\\experiment_results'
save_files_path = '.\\experiment_results'
hv_file_name = 'mean_hv_convergence.txt'
igd_file_name = 'mean_igd_convergence.txt'

generate_convergence_plot(hv_file_name, results_path_1, algorithm_1, results_path_2, algorithm_2)
generate_convergence_plot(igd_file_name, results_path_1, algorithm_1, results_path_2, algorithm_2)