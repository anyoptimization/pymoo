import copy

import numpy as np
import pygmo
import scipy
from scipy import stats
from scipy.spatial.distance import pdist

from algorithms.nsga import NSGA
from measures.igd import IGD
from metamodels.gpflow_metamodel import GPFlowMetamodel
from metamodels.selection_error_probablity import calc_sep
from model.algorithm import Algorithm
from model.individual import Individual
from operators.lhs_factory import LHS
from operators.random_spare_factory import RandomSpareFactory
from rand.default_random_generator import DefaultRandomGenerator
from run_analyse_nsao import show_objective_space
from util.misc import get_x, get_f, evaluate, perpendicular_dist, get_g, calc_metamodel_goodness, calc_mse, \
    get_hist_from_pop, get_front, get_front_by_index
from util.rank_and_crowding import RankAndCrowdingSurvival
from vendor.pygmo_solver import solve_by_moad


class NSAO(Algorithm):
    def __init__(self,
                 n_initial_DOE,
                 DOE_per_epoch=4,
                 reference_directions=None,
                 factory=RandomSpareFactory()
                 ):
        self.reference_directions = reference_directions
        self.DOE_per_epoch = DOE_per_epoch
        self.factory = factory
        self.n_initial_DOE = n_initial_DOE

    def solve_(self, problem, evaluator,
               rnd=DefaultRandomGenerator(),  # random generator to be used
               ):

        self.problem = problem

        # objects for artificial calculations on the metamodel
        X = LHS().sample(self.n_initial_DOE, problem.xl, problem.xu)
        opt_x, opt_f, opt_g = NSGA().solve(problem, 10000, 1)

        pop = [Individual(x) for x in X]
        evaluate(evaluator, problem, pop)

        # random points just for MSE calculation on test problem
        random_for_mse = LHS().sample(100, problem.xl, problem.xu)

        # uncertainty for the prediction of each objective
        uncertainty = 0.0 * np.ones(problem.n_obj)

        gen = 0

        while evaluator.has_next():
            gen += 1

            print("----------------------")
            print("Epoch %s" % gen)
            print("----------------------")
            print("Evaluations: %s" % evaluator.counter)

            # learn a metamodel on the data
            metamodel = GPFlowMetamodel()
            metamodel.fit(get_x(pop), get_f(pop))
            print(metamodel.parameters)

            # this can only be done for test problems of course
            mse_true = calc_metamodel_goodness(problem, metamodel, X=random_for_mse, func=calc_mse)
            sep_true = calc_metamodel_goodness(problem, metamodel, X=random_for_mse, func=calc_sep)
            mse_opt = calc_metamodel_goodness(problem, metamodel, X=opt_x, func=calc_mse)
            sep_opt = calc_metamodel_goodness(problem, metamodel, X=opt_x, func=calc_sep)
            print("A-MSE: %s, A-SEP:%s, AOPT-MSE: %s, AOPT-SEP: %s" % (mse_true, sep_true, mse_opt, sep_opt))

            print("Uncertainty Weights: %s" % uncertainty)

            x_hat, f_hat = self.get_front_by_moead(metamodel, pop, uncertainty)
            print("Non-Dominated Solutions on Metamodel: %s" % f_hat.shape[0])

            # remove solutions that are dominated by any current DOE
            merge = np.concatenate((get_f(pop), f_hat), axis=0)
            non_dom = [int(i - len(pop)) for i in get_front_by_index(merge) if i >= len(pop)]
            x_hat = x_hat[non_dom, :]
            f_hat = f_hat[non_dom, :]

            # select and evaluate expensively
            n_selected = min(self.DOE_per_epoch, evaluator.count_left())
            selected = self.select_from_front_max_dist(get_front(get_f(pop)), f_hat, n_selected)
            print("Selected: %s " % len(selected))

            # evaluate expensively the selected points
            DOE = [Individual(x_hat[s, :]) for s in selected]
            evaluate(evaluator, problem, DOE)

            # check the deviation from metamodel to evaluation
            mse_selected = [calc_mse(get_f(DOE)[:, i], f_hat[selected, i]) for i in range(problem.n_obj)]
            sep_selected = [calc_sep(get_f(DOE)[:, i], f_hat[selected, i]) for i in
                            range(problem.n_obj)]
            print("Metamodel Goodness for selected: %s" % mse_selected)

            show_objective_space(evaluator.counter, get_f(pop), f_hat, get_f(DOE), f_hat[selected, :])

            evaluator.notify(
                {'snapshot': get_hist_from_pop(pop, len(pop)),
                 'n_evals': evaluator.counter,
                 'sep_true': sep_true,
                 'mse_true': mse_true,
                 'sep_opt': sep_opt,
                 'mse_opt': mse_opt,
                 'sep_selected': sep_selected,
                 'mse_selected': mse_selected,
                 'metamodels': [e.__class__.__name__ for e in metamodel.parameters],
                 'pop_f': get_f(pop),
                 'f': problem.evaluate(x_hat)[0],
                 'f_hat': f_hat,
                 'f_selected': get_f(DOE),
                 'f_selected_hat': f_hat[selected, :]
                 })

            pop.extend(DOE)

        print('Done')

        return get_x(pop), get_f(pop), get_g(pop)

    def get_front_by_moead(self, meta_model, pop, uncertainty):

        # optimize the problem on the metamodel
        meta_model_problem = copy.deepcopy(self.problem)

        # evaluation function for the algorithm to be optimized on the metamodel
        def evaluate_(x, f):
            f[:, :], std = meta_model.predict(x)

            for i in range(f.shape[0]):
                for j in range(f.shape[1]):
                    f[i, j] = stats.norm.interval(uncertainty[j], loc=f[i, j], scale=std[i, j])[0]

        meta_model_problem.func = evaluate_
        x, f, _ = solve_by_moad(meta_model_problem)

        return x, f

    def get_front_by_nsga_on_problem(self, meta_model, pop, uncertainty):

        # optimize the problem on the metamodel
        problem = copy.deepcopy(self.problem)

        # problem.n_constr = 1

        # evaluation function for the algorithm to be optimized on the metamodel
        def evaluate_(x, f):
            f[:, :], std = meta_model.predict(x)

            # for i in range(f.shape[0]):
            #    for j in range(f.shape[1]):
            #        f[i, j] = stats.norm.interval(uncertainty[j], loc=f[i, j], scale=std[i, j])[0]

        problem.func = evaluate_
        x, f, _ = NSGA().solve(problem, 10000, return_only_feasible=True, return_only_non_dominated=True)

        # f, std = meta_model.predict(x)

        return x, f

    def optimize_bi_objective(self):
        pass

    def select_max_crowding(self, f, f_hat, n):
        all = np.concatenate((f, f_hat), axis=0)
        crowding = RankAndCrowdingSurvival.calc_crowding_distance(all)[len(f):]
        sorted_index = sorted(range(len(crowding)), key=lambda x: -crowding[x])

        print("")

    def select_from_front_max_dist(self, f, f_hat, n):

        # if less than n provided return f_all
        if n > len(f_hat):
            return list(range(len(f_hat)))

        # normalize f_all values according to current front
        f_min = np.min(f, axis=0)
        f_max = np.max(f, axis=0)

        f_all = np.concatenate((f, f_hat), axis=0)
        f_all = (f_all - f_min) / (f_max - f_min)

        selected = []
        to_select = list(range(len(f), len(f) + len(f_hat)))
        given = list(range(len(f)))

        def add_to_selected(i):
            selected.append(i)
            to_select.remove(i)

        for i, f_all_min in enumerate(np.argmin(f_all, axis=0)):
            if f_all[f_all_min,i] < 0:
                add_to_selected(f_all_min)

        dist_matrix = scipy.spatial.distance.squareform(pdist(f_all))

        while len(selected) < n:

            # optimize distances on the front
            indices = np.array(range(len(f) + len(f_hat)))[to_select]
            min_dists_from_given_to_others = np.min(dist_matrix[to_select][:, (given + selected)], axis=1)
            max_of_min_dists = np.argmax(min_dists_from_given_to_others)
            selected_idx = indices[int(max_of_min_dists)]

            add_to_selected(selected_idx)



        return [s - len(f) for s in selected]

    def select_from_front_min_igd(self, pop, front, n):
        f = np.array([ind.f for ind in pop])
        igd_if_added = np.zeros(len(front))
        for i in range(len(front)):
            igd_if_added[i] = IGD(self.problem.pareto_front()).calc(np.concatenate((f, np.array([front[i].f])), axis=0))
        selected = np.argsort(igd_if_added)[:n]
        return [front[i] for i in selected]

    def select_from_front_by_ref_dirs(self, x, f, x_hat, f_hat, n):

        idx_non_dom = pygmo.fast_non_dominated_sorting(f)[0][0]
        non_dom_f = f[idx_non_dom, :]
        f = np.concatenate((non_dom_f, f_hat))

        # normalize the values
        f_min = np.min(f, axis=0)
        f_max = np.max(f, axis=0)

        f_norm = (f - f_min) / (f_max - f_min)

        # assign each point to reference directions
        assigned_to_ref_dir = np.zeros(len(f))
        dist_to_ref_dir = np.zeros(len(f))
        for i in range(len(f_norm)):
            dists = [perpendicular_dist(ref, f_norm[i, :]) for ref in self.reference_directions]
            assigned_to_ref_dir[i] = np.argmin(dists)
            dist_to_ref_dir[i] = np.min(dists)  # + np.linalg.norm(f_norm[i, :])

        # number of assigned points to ref dir
        refs_count = np.zeros(len(self.reference_directions))

        # assign known DOE's
        for ref_idx in assigned_to_ref_dir[:len(non_dom_f)]:
            refs_count[int(ref_idx)] += 1

        assigned_to_ref_dir = assigned_to_ref_dir[len(non_dom_f):]
        dist_to_ref_dir = dist_to_ref_dir[len(non_dom_f):]

        # select new points to evaluate by looking at the min distance to least assigned reference direction
        selected = []
        while len(selected) < n and not np.all(np.isinf(refs_count)):

            # reference direction with least assignments - if tie choose randomly
            min_refs_count = np.where(refs_count == np.min(refs_count))[0]
            ref_dir = min_refs_count[np.random.randint(0, len(min_refs_count), 1)[0]]

            # points assigned to that reference direction and sort by dist
            points = np.where(assigned_to_ref_dir == ref_dir)[0]
            points = sorted(points.tolist(), key=lambda x: dist_to_ref_dir[x])

            # add a new point that is not selected yet
            point_added = False
            for p in points:
                if p not in selected:
                    selected.append(p)
                    point_added = True
                    break

            # new point to ref dir was added
            if point_added:
                refs_count[ref_dir] += 1.0
            # no point was found -> look for other ref points later
            else:
                refs_count[ref_dir] = np.inf

        return selected
