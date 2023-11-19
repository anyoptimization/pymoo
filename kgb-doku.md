KGB-DMOEA (Knowledge-Guided Bayesian Dynamic 
Multi-Objective Evolutionary Algorithm) Overview 
KGB-DMOEA is a sophisticated evolutionary algorithm 
for dynamic multi-objective optimization problems 
(DMOPs). It employs a knowledge-guided Bayesian 
classification approach to adeptly navigate and 
adapt to changing Pareto-optimal solutions in 
dynamic environments. This algorithm utilizes past 
search experiences, distinguishing them as 
beneficial or non-beneficial, to effectively direct 
the search in new scenarios. Key Features
	• Knowledge Reconstruction-Examination 
	(KRE): Dynamically re-evaluates historical 
	optimal solutions based on their relevance 
	and utility in the current environment. • 
	Bayesian Classification: Employs a Naive 
	Bayesian Classifier to forecast 
	high-quality initial populations for new 
	environments. • Adaptive Strategy: 
	Incorporates dynamic parameter adjustment 
	for optimized performance across varying 
	dynamic contexts.
Parameters • perc_detect_change (float, optional): 
	Proportion of the population used to detect 
	environmental changes. Default: 0.1. • 
	perc_diversity (float, optional): 
	Proportion of the population allocated for 
	introducing diversity. Default: 0.3. • 
	c_size (int, optional): Cluster size. 
	Default: 13. • eps (float, optional): 
	Threshold for detecting changes. Default: 
	0.0. • ps (dict, optional): Record of 
	historical Pareto sets. Default: {}. • 
	pertub_dev (float, optional): Deviation for 
	perturbation in diversity introduction. 
	Default: 0.1. • save_ps (bool, optional): 
	Option to save Pareto set data. Default: 
	False.
Methods • __init__(**kwargs): Initializes the 
	KGB-DMOEA algorithm with the provided 
	parameters. • 
	knowledge_reconstruction_examination(): 
	Implements the KRE strategy. • 
	naive_bayesian_classifier(pop_useful, 
	pop_useless): Trains the Naive Bayesian 
	Classifier using useful and useless 
	populations. • add_to_ps(): Incorporates 
	the current Pareto optimal set into the 
	historical Pareto set. • 
	predicted_population(X_test, Y_test): 
	Constructs a predicted population based on 
	classifier outcomes. • 
	calculate_cluster_centroid(solution_cluster): 
	Calculates the centroid for a specified 
	solution cluster. • check_boundaries(pop): 
	Ensures all population solutions stay 
	within defined problem boundaries. • 
	random_strategy(N_r): Generates a random 
	population within the bounds of the 
	problem. • diversify_population(pop): 
	Introduces diversity to the population. • 
	_advance(**kwargs): Progresses the 
	optimization algorithm by one iteration.
Usage Example from pymoo.algorithms.moo.kgb import 
KGB
# Define your problem
problem = ...
# Initialize KGB-DMOEA with specific parameters
algorithm = KGB( perc_detect_change=0.1, 
    perc_diversity=0.3, c_size=13, eps=0.0, ps={}, 
    pertub_dev=0.1, save_ps=False
)
# Execute the optimization
res = minimize(problem, algorithm, ...) References
	1.	Yulong Ye, Lingjie Li, Qiuzhen Lin, Ka-Chun Wong, Jianqiang Li, Zhong Ming. “A knowledge guided Bayesian classification for dynamic multi-objective optimization”. Knowledge-Based Systems, Volume 251, 2022. Link to the paper
