# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 16:45:11 2017

@author: Dhebar
"""

#....trials....


import time
s_time = time.time()
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import nsga2_classes
import nsga_funcs as nf
import pandas as pd
import trial_functs as tf
import trial_params as tp

import matplotlib

n_obj = 3
k = 5
dim = n_obj + k - 1

x = np.random.rand(dim)
##..DTLZ1...(3obj)
#	for (j=nobj-1,g=0.0; j<nreal; j++)
#      {
#          g += 100.0*(1.0 + (xreal[j]-0.5)*(xreal[j]-0.5) - cos(20.0*PI*(xreal[j]-0.5)));
#      }
g = 0
for j in range(n_obj-1,dim):
    g += 100.0*(1.0 + math.pow((x[j]-0.5),2) - math.cos(20.0*math.pi*(x[j] - 0.5)))
f_1 = 0.5*np.prod(x[:n_obj-1])*(1 + g)
f_2 = 0.5*np.prod(x[:n_obj-2])*(1.0 - x[n_obj - 2])*(1+g)
f_3 = 0.5*(1.0 - x[0])*(1 + g)

