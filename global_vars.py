# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 21:21:15 2017

@author: Dhebar
"""
import nsga2_classes
import numpy as np

#...global vars....
def declare_global_vars():
    global params
    params = nsga2_classes.Parameters()
    global ref_pts
    ref_pts = []
    global ref_pts_list
    ref_pts_list = []
    global ref_lines
    global parent_pop, child_pop
    parent_pop = []
    child_pop = []
    global ideal_pt
    ideal_pt = np.array([])
    global z_max
    
def declare_cons():
    global EPS
    EPS = 1.0e-14
    global EPS_weight
    EPS_weight = 0.001
    global weight_vectors_axes
    
    
