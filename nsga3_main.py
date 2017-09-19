# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 13:49:02 2017

@author: yddhebar
"""

#....NSGA-III....
import nsga2_classes
import nsga3_funcs as nf
import parameter_inputs
import global_vars
import numpy as np

global_vars.declare_global_vars()
global_vars.params = parameter_inputs.input_parameters()
global_vars.declare_cons()
nf.das_dennis_pts()

n_ref_lines = len(global_vars.ref_pts_list)
global_vars.ref_pts_list = np.array(global_vars.ref_pts_list)

global_vars.ref_lines = []
for i in range(n_ref_lines):
    global_vars.ref_lines.append(nsga2_classes.RefLine())
    global_vars.ref_lines[-1].direction = np.array(global_vars.ref_pts_list[i])

global_vars.ref_lines = np.array(global_vars.ref_lines)

global_vars.ideal_pt = np.array([float('inf')]*global_vars.params.n_obj)
global_vars.z_max = np.array([[0.0]*global_vars.params.n_obj]*global_vars.params.n_obj)
global_vars.weight_vectors_axes = np.array([[global_vars.EPS_weight]*global_vars.params.n_obj]*global_vars.params.n_obj)
for i in range(global_vars.params.n_obj):
    global_vars.weight_vectors_axes[i,i] = 1

print 'gen = %d'%(1)
global_vars.parent_pop = nf.initialize_popNSGA3(global_vars.params.pop_size)
nf.compute_fitness_pop(global_vars.parent_pop)


for gen_no in range(global_vars.params.max_gen):
    global_vars.child_pop = nf.selection(global_vars.parent_pop)
    nf.mutation_pop(global_vars.child_pop)
    
    mixed_pop = global_vars.parent_pop + global_vars.child_pop
#    F = nf.fill_nondominated_sort(mixed_pop)
    