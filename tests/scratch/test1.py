# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:26:16 2019

@author: omar.elfarouk
"""
# from pyomo.environ import *
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.nsga3 import NSGA3
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_termination
from pymoo.util import plotting
import autograd.numpy as anp
import numpy as np
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
import numpy as np
import math

# from scipy.optimize import rosen, rosen_der
# from ipopt import minimize_ipopt


# Factors for the third party in the supply chain%%
U_Demand = 12000  # normrnd(mn,std)%Monte carlo simulation mean 12000 unit%
alpha = 0.1  # Percentage of rate of return of products from third party%
lamda = alpha * U_Demand
miu = 0.1  # Probability of havieng a returned product in a good working condition%
gamma = 0.7  # Probability of having a rerurned part after disassembly with good working condition%

Q_TP = (lamda * (miu)) + (lamda * (1 - miu) * gamma)  # Quantity of the third party%

std = 200  # 10000 runslamda and %
# for daikin australia
pd_1 = 600  # round(random('Normal',600,24.49));q
pd_2 = 60  # round(random('Normal',60,7.749));
Z_var = U_Demand - pd_1 - pd_2
# Transportation Costs#
TS_s = 5000  # Transportation cost for the supplier(From alexandria road to downtown cairo)%
TS_m = 5000  # Transportation cost for the manufacturer(Assumed to be almost fixed due to practicallity)%
TS_d = 5000  # Transportation cost for the distributer%
TS_rt = 5000  # Transportation cost for the retailer%
TS_tp = 5000  # Transportation cost for the third party%
# collection Costs%%
C_tp = 5.1  # collection cost of recovering used product from the customer%
# facility opening Costs%%
F_rt = 10000000  # facility opening cost for the recovery center(Assumed to be 10 million  Egyptian pound)%
# Ordering Costs%%
S_s = 5.1
S_ms = 58.956
S_m1 = 700
S_m2 = 800
S_m3 = 850
S_d = 173.4
S_r = 204
S_tp = 42.5
# Holding Costs%%
H_s = 50.126
H_ms = 589.56
H_m = 1473.9
H_dr = 1734
H_rt = 2040
H_tp = 425.9571
# Production Rates%%
P_m1 = 200  # Production Rates assumed to be 200 unit per day for the power plant %%
P_m2 = 210
P_m3 = 220
# U_Demand = 400000 #Demand rate is asumed to be 400,000 unit per month%
P_m = P_m1 + P_m2 + P_m3  # Production rate of the manufacuter
# i_m #conunting of manufacturer%
# i_mp
# i_d   #Counting of Distributer
##Factors for the third party in the supply chain##
alpha = 0.1  # Percentage of rate of return of products from third party%
lamda = (alpha * U_Demand)
miu = 0.1  # Probability of havieng a returned product in a good working condition%
gamma = 0.7  # Probability of having a rerurned part after disassembly with good working condition%

Q_TP = (lamda * (miu)) + (lamda * (1 - miu) * gamma)  # Quantity of the third party%
# Values of supplied chain quantities
n_s = 5
n_m = 1  # 1:2
n_d = 1


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=12,
                         n_obj=3,
                         n_constr=16,
                         elementwise_evaluation=True,
                         xl=anp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                         xu=anp.array([12800000000000, 12800000000000, 12800000000000, 12800000000000, 12800000000000,
                                       12800000000000, 12800000000000, 12800000000000, 12800000000000, 12800000000000,
                                       12800000000000, 12800000000000]))

    def _evaluate(self, x, out, *args, **kwargs):  # Defining the Objective and constrains#
        Q_rt1 = x[0]  # quantity of the retailer in the forward cycle
        Q_rt2 = x[1]  # quantity of the retailer in the forward cycle
        Q_rt3 = x[2]  # quantity of the retailer in the forward cycle
        Q_d1 = x[3]  # Quantity of the distributer
        Q_d2 = x[4]  # Quantity of the distributer
        Q_d3 = x[5]  # Quantity of the distributer
        Q_m1 = x[6]  # Quantity of the Manufacturer
        Q_m2 = x[7]  # Quantity of the Manufacturer
        Q_m3 = x[8]  # Quantity of the Manufacturer
        Q_s1 = x[9]  # Quantity of Supplied Parts
        Q_s2 = x[10]  # Quantity of Supplied Parts
        Q_s3 = x[11]  # Quantity of Supplied Parts
        t_r = (U_Demand) / (x[0])  # Cycle time of the supply chain# #cycle time of the retailer
        t_d = n_d * t_r  # cycle time of the Distribiter
        t_m = (n_m * n_d * t_r)  # cycle time of the Manufacturer
        t_s = n_s * n_m * n_d * t_r  # cycle time of the supplier
        t_tp = t_s  # cycle time of the third party
        S_jfS = 30  # Job Index factor number of fixed jobs at the supplier assumed to be 30 fixed employees %
        S_jfM = 30  # Job index for the number of fixed jobs by Mamufacturer assumed to be 30 fixed employees %
        S_jfD = 30  # Job index for the number of fixed jobs by distributer assumed to be 30 fixed employees%
        S_jfRT = 30  # Job index for the number of fixed jobs by retialer assumed to be 30 fixed employees%
        S_jfTP = 20  # Job index for the number of fixed jobs by third party recovery assumed to be 20 fixed employees%
        S_jvS = 270  # Job Index factor number of variable jobs at the supplier assumed to be 270 workers per facility%
        S_jvM = 270  # Job index for the number of variable jobs by Mamufacturer  270 workers per facility%
        S_jvD = 270  # Job index for the number of variable jobs by distributer  270 workers per facility%
        S_jvRT = 270  # Job index for the number of variable jobs by retialer  270 workers per facility%
        S_jvTP = 100  # Job index for the number of variable jobs by third party recovery  100 workers per facility%
        S_u = 20  # Employee satisfaction factor of the refurbrished parts for the third party disassembler%
        S_rt = 30  # Customer satisfaction factor of the refurbrished parts%
        S_ds = 5  # Number of lost days at work% # Number of lost days from injuries or work damage at the suppliers / month%
        S_dm = 5  # Number of lost days from injuries or work damage at the manufactuer%
        S_dd = 5  # Number of lost days from injuries or work damage at the distributer%
        S_drt = 5  # Number of lost days from injuries or work damage at the retailer%
        S_dtp = 5  # Number of lost days from injuries or work damage at the third party%
        # Enviromental Aspect of the supply chain (Emissions calculated from carbon footprint)%
        E_q = 10  # Emission factor from production line
        E_tp = 10  # Emission from wastes removal%
        # Transportation emission cost%
        E_ts = 20  # Emission from Transportation made by the supplier%
        E_tm = 20  # Emission from Transportation made by the manufacturer%
        E_td = 20  # Emission from Transportation made by the distributer%
        E_trt = 20  # Emission from Transportation made by the retailer%
        E_ttp = 20  # Emission from Transportation made by the third party%
        i_s = 1
        i_ss = np.arange(i_s, n_s + 1, 1)
        tc_s1 = list(range(i_s, n_s + 1))
        for i_s in i_ss:
            tc_s1 = np.sum(((i_ss) / n_s) * Q_s1 * t_s)
            i_s = i_s + 1  # Adding value of Supplier integer#
        tc_s4 = (tc_s1)

        TC_s1 = (S_s * (1 / (n_s * t_s))) + (
                    ((H_s + TS_s) / (n_s * (t_s))) * tc_s4)  # cost of the supplier for component 1%

        i_s = 1  # starting of the loop#
        i_ss = np.arange(i_s, n_s + 1, 1)
        # for w1 in w11:
        tc_s2 = list(range(i_s, n_s + 1))
        for i_s in i_ss:
            tc_s2 = np.sum((i_ss / n_s) * Q_s2 * t_s)  # ((x(11)) +Q_TP#
            i_s = i_s + 1  # Adding value of Supplier integer
        tc_s5 = (tc_s2)

        TC_s2 = (S_s * (1 / (n_s * t_s))) + (((H_s + TS_s) / (n_s * (t_s))) * tc_s5)

        i_s = 1  # starting of the loop#
        tc_s3 = list(range(i_s, n_s + 1))
        for i_s in i_ss:
            tc_s3 = np.sum((i_ss / n_s) * Q_s3 * t_s)  # ((x(12)+ Q_TP))%
            i_s = i_s + 1  # Adding value of Supplier integer (No addition for Q_TP )%
        tc_s6 = tc_s3
        TC_s3 = (S_s * (1 / (n_s * t_s))) + (
                    ((H_s + TS_s) / (n_s * (t_s))) * tc_s6)  # cost of the supplier for component 3%
        i_m = 1  # starting of the loop#
        i_mm = np.arange(i_m, n_m + 1, 1)
        # for w1 in w11:
        tc_m2 = list(range(i_m, n_m + 1))
        for i_m in i_mm:
            tc_m1 = np.arange(1, n_m, 1)  # Defining range with starting and ending point
            tc_m2 = np.sum((1 - ((i_mm) / (n_m))) * ((Q_m1) + Q_TP))  # Defining range with start & ending point#
            i_m = i_m + 1  # Adding value of manufacturer integer#
        tc_m3 = (tc_m2)
        tc_s7 = np.arange(1, n_s, 1)
        # Total cost of manufacturer#
        tc_m = sum(tc_m1)
        tc_s8 = sum(tc_s7)
        TC_m = (H_m * ((0.5 * (Q_m1 ** 2) * (1 / P_m1)) \
                       + (tc_m * (Q_m1 * t_m * (1 / (n_m ** 2)))))) \
               + ((S_m1 + TS_m) * (1 / t_m)) + ((S_ms + TS_tp) * (1 / t_s)) \
               + (H_ms * (1 / t_s) * (((((Q_s1 + Q_TP) * Q_m1) / P_m1)) \
                                      + (tc_s8 * (((Q_s1) + Q_TP) / n_s) * (t_m - (Q_m1 / P_m1)))))

        TC_m2 = (H_m * ((0.5 * (Q_m2 ** 2) * (1 / P_m2)) \
                        + (tc_m * (Q_m2 * t_m * (1 / (n_m ** 2)))))) \
                + ((S_m2 + TS_m) * (1 / t_m)) + ((S_ms + TS_tp) * (1 / t_s)) \
                + (H_ms * (1 / t_s) * (((((Q_s2 + Q_TP) * Q_m2) / P_m2)) \
                                       + (tc_s8 * (((Q_s2) + Q_TP) / n_s) * (t_m - (Q_m2 / P_m2)))))
        TC_m3 = (H_m * ((0.5 * (Q_m3 ** 2) * (1 / P_m3)) \
                        + (tc_m * (Q_m3 * t_m * (1 / (n_m ** 2)))))) \
                + ((S_m3 + TS_m) * (1 / t_m)) + ((S_ms + TS_tp) * (1 / t_s)) \
                + (H_ms * (1 / t_s) * (((((Q_s3 + Q_TP) * Q_m3) / P_m3)) \
                                       + (tc_s8 * (((Q_s3) + Q_TP) / n_s) * (t_m - (Q_m3 / P_m3)))))

        i_d = 1
        i_dd = np.arange(i_d, n_d + 1, 1)
        # for w1 in w11:
        tc_d1 = list(range(i_d, n_d + 1))
        tc_d2 = list(range(i_d, n_d + 1))
        tc_d3 = list(range(i_d, n_d + 1))
        for i_d in i_dd:
            tc_d1 = np.sum(((i_dd) / (n_d)) * (Q_d1))  # Cost of the Distributer for Product 1%%
            tc_d2 = np.sum(((i_dd) / (n_d)) * (Q_d2))  # Cost of the Distributer for Product 2%%
            tc_d3 = np.sum(((i_d) / (n_d)) * (Q_d3))  # Cost of the Distributer for Product 3%%
            i_d = i_d + 1

        tc_d_f = (tc_d1) + (tc_d2) + (tc_d3)
        TC_d = (H_dr * (tc_d_f / n_d)) + (
                    (S_d + TS_d) * (1 / t_d))  # Total cost of the distributer of the supply chain%
        # Total cost of retailer

        TC_rt = (H_rt * ((Q_rt1) / 2)) + ((S_r + TS_rt) * (1 / t_r))  # Cost of the retailer%%
        TC_rt2 = (H_rt * (Q_rt2 / 2)) + ((S_r + TS_rt) * (1 / t_r))  # Cost of the retailer for product 2%%
        TC_rt3 = (H_rt * ((Q_rt3) / 2)) + ((S_r + TS_rt) * (1 / t_r))  # Cost of the retailer for product 3%%
        # Total cost of third party recovery
        TC_tp = ((H_tp / 2) * Q_TP) + ((S_tp + TS_tp) * (1 / t_tp))
        S_jfS = 30  # Job Index factor number of fixed jobs at the supplier assumed to be 30 fixed employees %
        S_jfM = 30  # Job index for the number of fixed jobs by Mamufacturer assumed to be 30 fixed employees %
        S_jfD = 30  # Job index for the number of fixed jobs by distributer assumed to be 30 fixed employees%
        S_jfRT = 30  # Job index for the number of fixed jobs by retialer assumed to be 30 fixed employees%
        S_jfTP = 20  # Job index for the number of fixed jobs by third party recovery assumed to be 20 fixed employees%
        S_jvS = 270  # Job Index factor number of variable jobs at the supplier assumed to be 270 workers per facility%
        S_jvM = 270  # Job index for the number of variable jobs by Mamufacturer  270 workers per facility%
        S_jvD = 270  # Job index for the number of variable jobs by distributer  270 workers per facility%
        S_jvRT = 270  # Job index for the number of variable jobs by retialer  270 workers per facility%
        S_jvTP = 100  # Job index for the number of variable jobs by third party recovery  100 workers per facility%
        S_u = 20  # Employee satisfaction factor of the refurbrished parts for the third party disassembler%
        S_rt = 30  # Customer satisfaction factor of the refurbrished parts%
        # Number of lost days at work%
        S_ds = 5  # Number of lost days from injuries or work damage at the suppliers / month%
        S_dm = 5  # Number of lost days from injuries or work damage at the manufactuer%
        S_dd = 5  # Number of lost days from injuries or work damage at the distributer%
        S_drt = 5  # Number of lost days from injuries or work damage at the retailer%
        S_dtp = 5  # Number of lost days from injuries or work damage at the third party%
        # Enviromental Aspect of the supply chain (Emissions calculated from carbon footprint)%
        E_q = 10  # Emission factor from production line
        E_tp = 10  # Emission from wastes removal%
        # Transportation emission cost%
        E_ts = 20  # Emission from Transportation made by the supplier%
        E_tm = 20  # Emission from Transportation made by the manufacturer%
        E_td = 20  # Emission from Transportation made by the distributer%
        E_trt = 20  # Emission from Transportation made by the retailer%
        E_ttp = 20  # Emission from Transportation made by the third party%
        # Cycle time%

        EQO = TC_s1 + TC_s2 + TC_s3 + TC_m + TC_m2 + TC_m3 + TC_d + TC_rt \
              + TC_rt2 + TC_rt3 + TC_tp
        #       Economical aspect#
        LSC = (S_jfS + S_jfM + S_jfD + S_jfRT + S_jfTP) \
              + ((S_jvS * Q_s1) + (S_jvD * Q_d1) + (S_jvM * Q_m1) \
                 + (S_jvRT * Q_rt1) + (S_jvTP * Q_TP)) \
              + (S_u * (U_Demand)) + (S_rt * Q_rt1) - (S_ds * Q_s1) \
              + (S_dd * Q_d1) + (S_dm * Q_m1) + (S_drt * Q_rt1) \
              + (S_dtp * Q_TP)  # Social aspect equation%

        ESC = (E_q * (Q_s1 + Q_d1 + Q_m1 + Q_rt1)) \
              + (E_ts * (1 / t_s)) + (E_td * (1 / t_d)) \
              + (E_tm * (1 / t_m)) + (E_trt * (1 / t_r)) \
              + (E_ts * (1 / t_tp)) + (E_tp * Q_TP)  # Enviromental aspect

        w1 = 1
        w2 = 1
        w3 = 1
        f1 = EQO * w1
        f2 = LSC * w2
        f3 = ESC * w3

        g1 = -x[0] + U_Demand
        g2 = -x[1] + U_Demand
        g3 = -x[2] + U_Demand
        g4 = -f1 - f2 - f3
        g5 = -((x[9]) + Q_TP) + (n_s * x[6])
        g6 = -((x[10]) + Q_TP) + (n_s * x[7])
        g7 = -((x[11]) + Q_TP) + (n_s * x[8])
        g8 = (n_m * (x[3])) - x[6]
        g9 = (n_m * (x[4])) - x[7]
        g10 = (n_m * (x[5])) - x[8]
        g11 = -x[3] + (n_d * x[0])
        g12 = -x[4] + (n_d * x[1])
        g13 = -x[5] + (n_d * x[2])
        g14 = -x[0]
        g15 = -x[1]
        g16 = -x[2]

        out["F"] = anp.column_stack([f1, f2, f3])
        out["G"] = anp.column_stack([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, \
                                     g13, g14, g15, g16])


problem = MyProblem()

ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
algorithm = NSGA3(ref_dirs)

termination = get_termination("f_tol", tol=0.001, n_last=20, n_max_gen=1000, nth_gen=10)
termination = ("n_gen", 10000)

res = minimize(MyProblem(),
               algorithm,
               termination,
               seed=1,
               pf=problem.pareto_front(use_cache=False),
               save_history=True,
               verbose=True)

# res = minimize(objective,x0,algorithm,bounds=bnds,constraints=cons('n_gen', 200))
# minimize(objective,x0,method='COBYLA',bounds=bnds,constraints=cons)

plot = Scatter()
plot.add(res.F, color="red")
plot.show()
