# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 15:18:39 2017

@author: Dhebar
"""

#....nsga2 functions....
import nsga2_classes
import parameter_inputs
import random
import numpy as np
import sys
import math
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import global_vars
import test_problems

#global_vars.params = parameter_inputs.input_parameters()
cons = parameter_inputs.input_constants()

def initialize_pop(pop_size):
    #...initializes the population pool
    pop = []
    for i in range(pop_size):
        pop.append(nsga2_classes.Individual())
        pop[i].xreal = np.array([0.0]*global_vars.params.n_var)
        pop[i].fitness = np.array([0.0]*global_vars.params.n_obj)
        pop[i].address = i
        for j in range(global_vars.params.n_var):
            pop[i].xreal[j] = random.uniform(global_vars.params.Lbound[j],global_vars.params.Ubound[j])
    return pop
    
def initialize_popNSGA3(pop_size):
    pop = []
    for i in range(pop_size):
        pop.append(nsga2_classes.IndividualNSGA3())
        pop[i].xreal = np.array([0.0]*global_vars.params.n_var)
        pop[i].fitness = np.array([0.0]*global_vars.params.n_obj)
        pop[i].address = i
        for j in range(global_vars.params.n_var):
            pop[i].xreal[j] = random.uniform(global_vars.params.Lbound[j],global_vars.params.Ubound[j])
        
    return pop
        
def compute_fitness_pop(pop):
    for i in range(len(pop)):
        test_problems.compute_fitness_ind(pop[i])
    return

def dominates(p,q):
    #...check if p dominates q
    #..returns 1 of p dominates q, -1 if q dominates p, 0 if both are non-dominated
    flag1 = 0
    flag2 = 0
    for i in range(global_vars.params.n_obj):
        if p.fitness[i] < q.fitness[i]:
            flag1 = 1
        else:
            if p.fitness[i] > q.fitness[i]:
                flag2 = 1
        
    if (flag1 == 1) and (flag2 == 0):
        return 1
    else:
        if (flag1 == 0) and (flag2 == 1):
            return -1
        else:
            return 0

#def dominates(p,q):
#    #...check if p dominates q
#    for i in range(global_vars.params.n_obj):
#        if p.fitness[i] > q.fitness[i]:
#            return 0
#    return 1

def sort_wrt_obj(f,j):
    obj_values = []
    for p in f:
        obj_values.append(p.fitness[j])
        
    r = range(len(f))
    g = [x for _,x in sorted(zip(obj_values,r))]
    return g,max(obj_values),min(obj_values)
           
def crowding_distance(F):
    #....
    for f in F:
        f_len = len(f)
        if (f_len <= 2):
            for p in f:
                p.crowding_dist = float('inf')
        else:
            for p in f:
                p.crowding_dist = 0.0
    
            for j in range(global_vars.params.n_obj):
                f_id, max_j, min_j = sort_wrt_obj(f,j)
                f[f_id[0]].crowding_dist = float("inf")
                #f[f_id[-1]].crowding_dist = float("inf")
                if max_j != min_j:
                    for i in range(1,f_len-1):
                        #if f[f_id[i]].crowding_dist == float('inf'):
                        #    f[f_id[i]].crowding_dist = 0
                        f[f_id[i]].crowding_dist = f[f_id[i]].crowding_dist + \
                        (f[f_id[i+1]].fitness[j] - f[f_id[i-1]].fitness[j])/(max_j - min_j)

    
    return

def cw_dist_inf_count(F):
    for f in F:
        cw_list = []
        for p in f:
            cw_list.append(p.crowding_dist)
        inf_count = cw_list.count(float('inf'))
        if inf_count > 2:
            print ('inf_count = %d')%inf_count
            #...count non_dominating members...
            count_dom = 0
            f_len = len(f)
            for i_id in range(f_len):               
                for j in range(i_id+1,f_len): 
                    if (abs(dominates(f[i_id],f[j]))):
                        count_dom += 1
            print 'count_dom = %d, f_len = %d, f[i](i = %d)'%(count_dom,f_len,F.index(f))
            return 1
    return 0
    
def assign_rank_crowding_distance(pop):
    F = []#...list of fronts...
    f = []
#    counter = 0
    for p in pop:
        p.S_dom = []
        p.n_dom = 0
        for q in pop:
            dom_p_q = dominates(p,q)
            if (dom_p_q == 1):
                p.S_dom.append(q)
            elif (dom_p_q == -1):
                p.n_dom = p.n_dom + 1
                
        if p.n_dom == 0:
            p.rank = 1
#            counter += 1
#            if counter > global_vars.params.pop_size:
#                break
            f.append(p)
        
    F.append(f)
    i = 0
    while (len(F[i]) != 0):
#        if counter > global_vars.params.pop_size:
#            break
        Q = []
        for p in F[i]:
            for q in p.S_dom:
                q.n_dom = q.n_dom - 1
                if q.n_dom == 0:
                    q.rank = i + 2
                    Q.append(q)
        i = i+1
        F.append(Q)
#        counter += len(Q)
#    
#    if counter <= global_vars.params.pop_size:  
    del(F[-1])
        
    crowding_distance(F)
    
    if cw_dist_inf_count(F):
        print 'in assign_rank_crowding_distance() routine'
        sys.exit()
    
    return F

def sort_wrt_crowding_dist(f):
    c_dist_vals = []
    for p in f:
        c_dist_vals.append(p.crowding_dist)
        
    r = range(len(f))
    f_new_id = [x for _,x in sorted(zip(c_dist_vals,r),reverse=True)]
    return f_new_id


def tour_select(ind1, ind2):
    #print 'doing binary tournament selection'
    dom_1_2 = dominates(ind1,ind2)
    if (dom_1_2 == 1):
        return(ind1)
    elif (dom_1_2 == -1):
        return(ind2)
    else:
        if (ind1.crowding_dist > ind2.crowding_dist):
            return(ind1)
        elif (ind2.crowding_dist > ind1.crowding_dist):
            return(ind2)
        elif (random.random() > 0.5):
            return(ind1)
        else:
            return(ind2)
    
    

def crossover(parent1,parent2):
    #child1 = copy.deepcopy(parent1)
    #child2 = copy.deepcopy(parent2)
    child1 = nsga2_classes.Individual()
    child2 = nsga2_classes.Individual()
    child1.xreal = np.array([0.0]*global_vars.params.n_var)
    child2.xreal = np.array([0.0]*global_vars.params.n_var)
    child1.fitness = np.array([0.0]*global_vars.params.n_obj)
    child2.fitness = np.array([0.0]*global_vars.params.n_obj)
    
    if (random.random() < global_vars.params.p_xover):
        for i in range(global_vars.params.n_var):
            if (random.random() < 0.5):
                if (abs(parent1.xreal[i] - parent2.xreal[i]) > cons.EPS):
                    if (parent1.xreal[i] < parent2.xreal[i]):
                        y1 = parent1.xreal[i]
                        y2 = parent2.xreal[i]
                    else:
                        y1 = parent2.xreal[i]
                        y2 = parent1.xreal[i]
                    
                    y_L = global_vars.params.Lbound[i]
                    y_U = global_vars.params.Ubound[i]
                    rand = random.random()
                    beta = 1.0 + (2.0*(y1 - y_L)/(y2 - y_L))
                    alpha = 2.0 - math.pow(beta,-(global_vars.params.eta_xover + 1))
                    if (rand <= (1.0/alpha)):
                        beta_q = math.pow((rand*alpha),
                                          (1.0/(global_vars.params.eta_xover+1.0)))
                    else:
                        beta_q = math.pow((1.0/(2.0 - rand*alpha)),
                                    (1.0/(global_vars.params.eta_xover+1.0)))
                
                    c1 = 0.5*((y1+y2) - beta_q*(y2-y1))
                    beta = 1.0 + (2.0*(y_U-y2)/(y2-y1))
                    alpha = 2.0 - math.pow(beta,-(global_vars.params.eta_xover+1.0))
                    if rand <= (1.0/alpha):
                        beta_q = math.pow((rand*alpha),
                                          (1.0/(global_vars.params.eta_xover+1.0)))
                    else:
                        beta_q = math.pow((1.0/(2.0-rand*alpha)),
                                          (1.0/(global_vars.params.eta_xover+1.0)))
                    c2 = 0.5*((y1+y2)+beta_q*(y2-y1))
                    if (c1 < y_L):
                        c1 = y_L
                    if (c2 < y_L):
                        c2 = y_L
                    if (c1 > y_U):
                        c1 = y_U
                    if (c2 > y_U):
                        c2 = y_U
                        
                    if (random.random() <= 0.5):
                        child1.xreal[i] = c2
                        child2.xreal[i] = c1
                    else:
                        child1.xreal[i] = c1
                        child2.xreal[i] = c2
                        
                else:
                    child1.xreal[i] = parent1.xreal[i]
                    child2.xreal[i] = parent2.xreal[i]
            else:
                child1.xreal[i] = parent1.xreal[i]
                child2.xreal[i] = parent2.xreal[i]
    else:
        child1.xreal = np.array(list(parent1.xreal))
        child2.xreal = np.array(list(parent2.xreal))
        
    return child1,child2
    
#Routine for tournament selection, it creates a new_pop from old_pop by 
#    performing tournament selection and the crossover  
def selection(old_pop):
    pop_size = len(old_pop)
    a1 = np.random.permutation(range(pop_size))
    a2 = np.random.permutation(range(pop_size))
    new_pop = [0]*pop_size
    for i in range(0,pop_size,4):
        parent1 = tour_select(old_pop[a1[i]],old_pop[a1[i+1]])
        parent2 = tour_select(old_pop[a1[i+2]],old_pop[a1[i+3]])
        new_pop[i], new_pop[i+1] = crossover(parent1,parent2)
        
        parent1 = tour_select(old_pop[a2[i]],old_pop[a2[i+1]])
        parent2 = tour_select(old_pop[a2[i+2]],old_pop[a2[i+3]])
        new_pop[i+2], new_pop[i+3] = crossover(parent1,parent2)
    
    
    #..address assignment...
    for i in range(len(new_pop)):
        new_pop[i].address = i
        
    return new_pop
        
def mutation_ind(ind):
    for j in range(global_vars.params.n_var):
        if (random.random() < global_vars.params.p_mut):
            y = ind.xreal[j]
            y_L = global_vars.params.Lbound[j]
            y_U = global_vars.params.Ubound[j]
            delta1 = (y-y_L)/(y_U - y_L)
            delta2 = (y_U - y)/(y_U - y_L)
            rnd = random.random()
            mut_pow = 1.0/(global_vars.params.eta_mut + 1.0)
            if (rnd <= 0.5):
                xy = 1.0 - delta1
                val = 2.0*rnd + (1.0 - 2.0*rnd)*(
                math.pow(xy,(global_vars.params.eta_mut + 1.0)))
                delta_q = math.pow(val,mut_pow) - 1.0
            else:
                xy = 1.0-delta2
                val = 2.0*(1.0-rnd)+2.0*(rnd-0.5)*(
                math.pow(xy,(global_vars.params.eta_mut+1.0)))
                delta_q = 1.0 - (math.pow(val,mut_pow))
                
            y = y + delta_q*(y_U - y_L)
            if (y < y_L):
                y = y_L
            if (y > y_U):
                y = y_U
                
            ind.xreal[j] = y
    return
        
def mutation_pop(pop):
    for p in pop:
        mutation_ind(p)
        
    return


#...upgrade, unoptimized....
def fill_nondominated_sort(mixed_pop):
    filtered_pop = []
    selected_fronts = assign_rank_crowding_distance(mixed_pop)
    counter = 0
    candidate_fronts = []
    for f in selected_fronts:
        candidate_fronts.append(f)
        counter += len(f)
        if counter > global_vars.params.pop_size:
            break
        
    n_fronts = len(candidate_fronts)
    #print 'n_fronts = %d'%n_fronts
    if n_fronts == 1:
        filtered_pop = []#candidate_fronts[0]
        #return filtered_pop
    else:
        for i in range(n_fronts - 1):
            filtered_pop.extend(candidate_fronts[i])
    
    n_pop_curr = len(filtered_pop)
    
    sorted_final_front_id = sort_wrt_crowding_dist(candidate_fronts[-1])
    
    if len(sorted_final_front_id) != len(set(sorted_final_front_id)):
        print 'ERRORRRRRRR... sorted not uniquely'
        sys.exit()
    
    for i in range(global_vars.params.pop_size - n_pop_curr):
        filtered_pop.append(candidate_fronts[-1][sorted_final_front_id[i]])
    
    #....count indivs with inf crowding dist
    
    if cw_dist_inf_count(candidate_fronts):
        print 'inside fill_nondominated_sort() routine, in candidate_fronts'
        sys.exit()
    

    return filtered_pop
    

def plot_pop(pop):
    #obj_vals = np.zeros([global_vars.params.pop_size,global_vars.params.n_obj
    obj_vals = np.zeros([global_vars.params.pop_size,global_vars.params.n_obj])
    
    for i in range(len(pop)):
        for j in range(global_vars.params.n_obj):
            obj_vals[i,j] = pop[i].fitness[j]
            
    if global_vars.params.n_obj == 2:
        f1 = obj_vals[:,0]
        f2 = obj_vals[:,1]
        plt.plot(f1,f2,'bo')
        plt.xlabel('f1')
        plt.ylabel('f2')
        plt.title(global_vars.params.prob_name)
    else:
        print 'sorry, cannot plot as n_obj > 2!!'
        
def write_final_pop_obj(pop,run):
    f_name = os.path.join('results',global_vars.params.prob_name + '_RUN' + str(run) + str('.out'))
    f = open(f_name,'w')
    for p in pop:
        f.write('%f \t %f\n'%(p.fitness[0],p.fitness[1]))
        
def generate_das_dennis(beta, depth):
    
    if (depth == (global_vars.params.n_obj - 1)):
        global_vars.ref_pts[depth] = beta/(1.0*global_vars.params.n_partitions) 
        global_vars.ref_pts_list.append(list(global_vars.ref_pts))
        print ('\n')
        return
    
    for i in range(beta+1):
        global_vars.ref_pts[depth] = 1.0*i/(1.0*global_vars.params.n_partitions)
        generate_das_dennis(beta - i, depth + 1)
        
def das_dennis_pts():
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
        