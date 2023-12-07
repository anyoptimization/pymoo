import numpy as np

from pymoo.core.problem import Problem
import pymoo.algorithms.soo.nonconvex.ga
import matplotlib.pyplot as plt


# Notes: 
# I'm using the suffix _vf to denote variables used in the value-function optimization 
# This is to avoid confuse it with variables being optimized in the main function 

# Input 1: A list of non-dominated points 
# Input 2: The ranking of the given non-dominated points 
# Input 3: constraint function for optimizing the value func
# Input 4: the skeleton utility function that we're trying to optimize
def create_vf(P, ranks, cFunc, utilFunc): 

    # Combine rankings and objective functions into one matrix 
    # The first O columns are the objective values, and the O + 1 column is the ranking

    X = np.hstack((P, np.array([ranks]).T))

    return lambda f_new:  np.sum(f_new)



def create_linear_vf(P, ranks): 

    return lambda f_new:  np.sum(f_new)


def linear_vf(P, x): 

    return np.matmul(P, x.T).T


def plot_linear_vf(P, x_vf): 

    plt.scatter(P[:,0], P[:,1], marker=".", color="red", s=200 )
   
    for i in range(np.size(P,0)):
        plt.annotate("P%d" % (i+1), (P[i,0], P[i,1]))

    # TODO pull 0 and 8 from min/max P 
    x,y = np.meshgrid(np.linspace(0, 8, 1000), np.linspace(0, 8, 1500))

    z = linear_vf(np.stack((x,y), axis=2), x_vf)[0] 

    z = z.T

    values_at_P = []
    for p in range(np.size(P,0)):

        values_at_P.append(linear_vf(P[p,:], x_vf)[0])

    values_at_P.sort()

    plt.contour(x,y,z, levels=values_at_P)

    plt.show()


#def scipyMin(P, ranks): 




class OptimizeVF(Problem): 

    def __init__(self, P, ranks, vf):
       
        # One var for each dimension of the object space, plus epsilon 
        n_var_vf = np.size(P, 1) + 1

        # it has one inequality constraints for every pair of solutions in P
        n_ieq_c_vf = np.size(P,0) - 1
       
        xl_vf = [0.0] * n_var_vf 
        xu_vf = [1.0] * n_var_vf
        
        # upper/lower bound on the epsilon variable is -1000/1000
        xl_vf[-1] = -1000
        xu_vf[-1] = 1000

        # TODO start everything at 0.5

        # Add the rankings onto our objectives 
        self.P = np.hstack((P, np.array([ranks]).T))

        # Sort P by rankings in the last column
        self.P = self.P[self.P[:, -1].argsort()]

        self.vf = vf

        super().__init__(n_var_vf, n_obj=1, n_ieq_constr=n_ieq_c_vf, n_eq_constr=1, xl=xl_vf, xu=xu_vf)


       

    def _evaluate(self, x, out, *args, **kwargs):

        ## Objective function: 
        obj = OptimizeVF._obj_func(x)

        # The objective function above returns a negated version of epsilon 
        ep = -obj

        # maximize epsilon, or the minimum distance between each contour 
        out["F"] = obj

        ## Inequality
        # TODO for now, assuming there are no ties in the ranks

        ineq_func = self._build_ineq_constr()

        out["G"] = ineq_func(x)
            
        ## Equality constraint that keeps sum of x under 1
        out["H"] = OptimizeVF._eq_constr(x)

    def build_constr(self):

        ineq_constr_func = self._build_ineq_constr();

        return lambda x : np.append(ineq_constr_func(x), self._eq_constr(x))



    def _build_ineq_constr(self):

        ineq_func = lambda x : OptimizeVF._ineq_constr(x, self.P, self.vf)

        return ineq_func

    @staticmethod
    def _ineq_constr(x, P, vf):
        if len(x.shape) == 1:
            return OptimizeVF._ineq_constr_1D(x, P, vf)
        else: 
            return OptimizeVF._ineq_constr_2D(x, P, vf)


    @staticmethod
    def _ineq_constr_2D(x, P, vf):

        ep = np.column_stack([x[:,-1]]) 
        pop_size = np.size(x,0)

        G = np.ones((pop_size, np.size(P,0)-1))*-99

        # Pair-wise compare each ranked member of P, seeing if our proposed utility 
        #  function increases monotonically as rank increases
        for p in range(np.size(P,0) - 1):

           current_P = vf(P[[p],0:-1], x[:, 0:-1])
           next_P = vf(P[[p+1],0:-1], x[:, 0:-1])

           G[:,[p]] = -(current_P - next_P) + ep

        return G


    @staticmethod
    def _ineq_constr_1D(x, P, vf):

        ep = x[-1]

        G = np.ones((1, np.size(P,0)-1))*-99

        # Pair-wise compare each ranked member of P, seeing if our proposed utility 
        #  function increases monotonically as rank increases
        for p in range(np.size(P,0) - 1):

           current_P = vf(P[[p],0:-1], x[0:-1])
           next_P = vf(P[[p+1],0:-1], x[0:-1])

           G[:,[p]] = -(current_P - next_P) + ep

        return G

    @staticmethod
    def _obj_func(x): 

        # check if x is a 1D array or a 2D array
        if len(x.shape) == 1:
            ep = x[-1]
        else: 
            ep = np.column_stack([x[:,-1]])

        return -ep


    # Constraint that states that the x values must add up to 1
    @staticmethod
    def _eq_constr(x): 

        if len(x.shape) == 1:
            eq_cons = sum(x[0:-1]) - 1
        else: 
            eq_cons = np.sum(x[:,0:-1],1, keepdims=True) - 1

        return eq_cons

     


