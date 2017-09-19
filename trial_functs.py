# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 09:51:37 2017

@author: yddhebar
"""

#...sample... trial file for functions....
import trial_params

def print_name():
    print trial_params.my_name
    trial_params.my_name = 'is changed'
    print trial_params.my_name
    
    return