#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 20:10:26 2018

@author: yashesh
"""

# ....das_dennis_points generation...
# import global_vars
import numpy as np
import matplotlib.pyplot as plt


def generate_das_dennis(beta, depth):
    if (depth == (n_obj - 1)):
        ref_pts[depth] = beta / (1.0 * n_partitions)
        ref_pts_list.append(list(ref_pts))
        print('\n')
        return

    for i in range(beta + 1):
        ref_pts[depth] = 1.0 * i / (1.0 * n_partitions)
        generate_das_dennis(beta - i, depth + 1)

    return


def das_dennis(scale, n_p):
    # ...inputs : scale and n_p(number of partitions)
    global n_obj
    n_obj = 3
    global n_partitions
    n_partitions = n_p
    global ref_pts
    global ref_pts_list
    n_partitions = n_p
    ref_pts_list = []
    ref_pts = [0] * n_obj
    generate_das_dennis(n_partitions, 0)

    ref_array1 = scale * np.array(ref_pts_list)
    return ref_array1


############## MAIN CODE ###################################################
ref_array = das_dennis(1.0, 12)

print(np.sum(ref_array, axis=1))

# ..plotting...
if n_obj == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = ref_array[:, 0]
    y = ref_array[:, 1]
    z = ref_array[:, 2]

    ax.scatter(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()



