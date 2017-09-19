# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:34:11 2017

@author: yddhebar
"""

#...generate das and dennis points....
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import parameter_inputs
import global_vars
import nsga2_classes


global_vars.ref_pts = [0]*global_vars.params.n_obj

        
generate_das_dennis(global_vars.params.n_partitions,0)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

ref_array = np.array(global_vars.ref_pts_list)

x = ref_array[:,0]
y = ref_array[:,1]
z = ref_array[:,2]

ax.scatter(x,y,z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()