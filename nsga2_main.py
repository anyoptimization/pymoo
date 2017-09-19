# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 14:40:18 2017

@author: Dhebar
"""

#....NSGA-II main code....

import nsga2_classes
import numpy as np
import random
import parameter_inputs
import nsga_funcs as nf
import copy
import matplotlib.pyplot as plt
import global_vars

global_vars.declare_global_vars()


random.seed()

global_vars.params = parameter_inputs.input_parameters()
cons = parameter_inputs.input_constants()

print 'pop_size = %d'%global_vars.params.pop_size

print 'initialize'

print 'gen = %d'%0
parent_pop = nf.initialize_pop(global_vars.params.pop_size)
nf.compute_fitness_pop(parent_pop)
nf.assign_rank_crowding_distance(parent_pop)

for run in range(global_vars.params.max_runs):
    for i in range(1,global_vars.params.max_gen):
        print 'gen = %d'%i
        child_pop = nf.selection(parent_pop)
        nf.mutation_pop(child_pop)
        nf.compute_fitness_pop(child_pop)
        mixed_pop = parent_pop + child_pop
        parent_pop = nf.fill_nondominated_sort(mixed_pop)


    nf.write_final_pop_obj(parent_pop,run+1)
nf.plot_pop(parent_pop)