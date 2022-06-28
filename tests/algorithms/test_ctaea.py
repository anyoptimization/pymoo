import numpy as np
import pytest

from pymoo.algorithms.moo.ctaea import (CADASurvival, RestrictedMating,
                                        comp_by_cv_dom_then_random)
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.problems.many import C1DTLZ1, C1DTLZ3, C3DTLZ4
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from tests.test_util import load_to_test_resource


@pytest.fixture
def ref_dirs():
   return np.loadtxt(load_to_test_resource('ctaea', 'weights.txt'))


@pytest.fixture
def evaluator():
    return Evaluator()


def test_association(ref_dirs, evaluator):
    problem = C1DTLZ3(n_var=12, n_obj=3)
    ca_x = np.loadtxt(load_to_test_resource('ctaea', 'c1dtlz3', 'case3', 'preCA.x'))
    CA = Population.new(X=ca_x)
    evaluator.eval(problem, CA)

    da_x = np.loadtxt(load_to_test_resource('ctaea', 'c1dtlz3', 'case3', 'preDA.x'))
    DA = Population.new(X=da_x)
    evaluator.eval(problem, DA)

    off_x = np.loadtxt(load_to_test_resource('ctaea', 'c1dtlz3', 'case3', 'offspring.x'))
    off = Population.new(X=off_x)
    evaluator.eval(problem, off)

    true_assoc = np.loadtxt(load_to_test_resource('ctaea', 'c1dtlz3', 'case3', 'feasible_rank0.txt'))
    true_niche = true_assoc[:, 1]
    true_id = true_assoc[:, 0]
    sorted_id = np.argsort(true_id)

    survival = CADASurvival(ref_dirs)
    mixed = Population.merge(CA, off)
    survival.ideal_point = np.min(np.vstack((DA.get("F"), mixed.get("F"))), axis=0)

    fronts = NonDominatedSorting().do(mixed.get("F"), n_stop_if_ranked=len(ref_dirs))
    I = np.concatenate(fronts)
    niche, _ = survival._associate(mixed[I])
    sorted_I = np.argsort(I)

    assert (niche[sorted_I] == true_niche[sorted_id]).all()


def test_update_ca(ref_dirs, evaluator):
    problem = C1DTLZ3(n_var=12, n_obj=3)
    ca_x = np.loadtxt(load_to_test_resource('ctaea', 'c1dtlz3', 'case3', 'preCA.x'))
    CA = Population.new(X=ca_x)
    evaluator.eval(problem, CA)

    da_x = np.loadtxt(load_to_test_resource('ctaea', 'c1dtlz3', 'case3', 'preDA.x'))
    DA = Population.new(X=da_x)
    evaluator.eval(problem, DA)

    off_x = np.loadtxt(load_to_test_resource('ctaea', 'c1dtlz3', 'case3', 'offspring.x'))
    off = Population.new(X=off_x)
    evaluator.eval(problem, off)

    post_ca_x = np.loadtxt(load_to_test_resource('ctaea', 'c1dtlz3', 'case3', 'postCA.x'))
    true_pCA = Population.new(X=post_ca_x)
    evaluator.eval(problem, true_pCA)

    survival = CADASurvival(ref_dirs)
    mixed = Population.merge(CA, off)
    survival.ideal_point = np.min(np.vstack((DA.get("F"), mixed.get("F"))), axis=0)

    pCA = survival._updateCA(mixed, len(ref_dirs))

    pX = set([tuple(x) for x in pCA.get("X")])
    tpX = set([tuple(x) for x in true_pCA.get("X")])
    assert pX == tpX

    problem = C1DTLZ1(n_var=9, n_obj=3)
    ca_x = np.loadtxt(load_to_test_resource('ctaea', 'c1dtlz1', 'preCA.x'))
    CA = Population.new(X=ca_x)
    evaluator.eval(problem, CA)

    da_x = np.loadtxt(load_to_test_resource('ctaea', 'c1dtlz1', 'preDA.x'))
    DA = Population.new(X=da_x)
    evaluator.eval(problem, DA)

    off_x = np.loadtxt(load_to_test_resource('ctaea', 'c1dtlz1', 'offspring.x'))
    off = Population.new(X=off_x)
    evaluator.eval(problem, off)

    post_ca_x = np.loadtxt(load_to_test_resource('ctaea', 'c1dtlz1', 'postCA.x'))
    true_pCA = Population.new(X=post_ca_x)
    evaluator.eval(problem, true_pCA)

    survival = CADASurvival(ref_dirs)
    mixed = Population.merge(CA, off)
    survival.ideal_point = np.min(np.vstack((DA.get("F"), mixed.get("F"))), axis=0)

    pCA = survival._updateCA(mixed, len(ref_dirs))

    pX = set([tuple(x) for x in pCA.get("X")])
    tpX = set([tuple(x) for x in true_pCA.get("X")])
    assert pX == tpX

    problem = C3DTLZ4(n_var=12, n_obj=3)
    ca_x = np.loadtxt(load_to_test_resource('ctaea', 'c3dtlz4', 'case1', 'preCA.x'))
    CA = Population.new(X=ca_x)
    evaluator.eval(problem, CA)

    da_x = np.loadtxt(load_to_test_resource('ctaea', 'c3dtlz4', 'case1', 'preDA.x'))
    DA = Population.new(X=da_x)
    evaluator.eval(problem, DA)

    off_x = np.loadtxt(load_to_test_resource('ctaea', 'c3dtlz4', 'case1', 'offspring.x'))
    off = Population.new(X=off_x)
    evaluator.eval(problem, off)

    post_ca_x = np.loadtxt(load_to_test_resource('ctaea', 'c3dtlz4', 'case1', 'postCA.x'))
    true_pCA = Population.new(X=post_ca_x)
    evaluator.eval(problem, true_pCA)

    survival = CADASurvival(ref_dirs)
    mixed = Population.merge(CA, off)
    survival.ideal_point = np.min(np.vstack((DA.get("F"), mixed.get("F"))), axis=0)

    pCA = survival._updateCA(mixed, len(ref_dirs))

    pX = set([tuple(x) for x in pCA.get("X")])
    tpX = set([tuple(x) for x in true_pCA.get("X")])
    assert pX == tpX


def test_update_da(ref_dirs, evaluator):
    problem = C1DTLZ3(n_var=12, n_obj=3)
    for i in range(2):
        ca_x = np.loadtxt(load_to_test_resource('ctaea', 'c1dtlz3', f'case{i + 1}', 'preCA.x'))
        CA = Population.new(X=ca_x)
        evaluator.eval(problem, CA)

        da_x = np.loadtxt(load_to_test_resource('ctaea', 'c1dtlz3', f'case{i + 1}', 'preDA.x'))
        DA = Population.new(X=da_x)
        evaluator.eval(problem, DA)

        off_x = np.loadtxt(load_to_test_resource('ctaea', 'c1dtlz3', f'case{i + 1}', 'offspring.x'))
        off = Population.new(X=off_x)
        evaluator.eval(problem, off)

        survival = CADASurvival(ref_dirs)
        mixed = Population.merge(CA, off)
        survival.ideal_point = np.min(np.vstack((DA.get("F"), mixed.get("F"))), axis=0)

        post_ca_x = np.loadtxt(load_to_test_resource('ctaea', 'c1dtlz3', f'case{i + 1}', 'postCA.x'))
        CA = Population.new(X=post_ca_x)
        evaluator.eval(problem, CA)

        Hd = Population.merge(DA, off)
        pDA = survival._updateDA(CA, Hd, 91)

        true_S1 = [151, 35, 6, 63, 67, 24, 178, 106, 134, 172, 148, 159, 41,
                   173, 145, 77, 62, 40, 127, 61, 130, 27, 171, 115, 52, 176,
                   22, 75, 55, 87, 36, 149, 154, 47, 78, 170, 90, 15, 53, 175,
                   179, 165, 56, 89, 132, 82, 141, 39, 32, 25, 131, 14, 72, 65,
                   177, 140, 66, 143, 34, 81, 103, 99, 147, 168, 51, 26, 70, 94,
                   54, 97, 158, 107, 29, 120, 50, 108, 157, 11, 85, 174, 80, 0,
                   95, 13, 142, 101, 156, 19, 8, 98, 20]

        true_S2 = [78, 173, 59, 21, 101, 52, 36, 94, 17, 20, 37, 96, 90, 129,
                   150, 136, 162, 70, 146, 75, 138, 154, 65, 179, 98, 32, 97,
                   11, 26, 107, 12, 128, 95, 170, 24, 171, 40, 180, 14, 44, 49,
                   43, 130, 23, 60, 79, 148, 62, 87, 56, 157, 73, 104, 45, 177,
                   74, 15, 152, 164, 28, 80, 113, 41, 33, 158, 57, 77, 34, 114,
                   118, 18, 54, 53, 145, 93, 115, 121, 174, 142, 39, 13, 105,
                   10, 69, 120, 55, 6, 153, 91, 137, 46]
        if i == 0:
            assert np.all(pDA == Hd[true_S1])
        else:
            assert np.all(pDA == Hd[true_S2])


def test_update(ref_dirs, evaluator):
    problem = C3DTLZ4(n_var=12, n_obj=3)
    ca_x = np.loadtxt(load_to_test_resource('ctaea', 'c3dtlz4', 'case2', 'preCA.x'))
    CA = Population.new(X=ca_x)
    evaluator.eval(problem, CA)

    da_x = np.loadtxt(load_to_test_resource('ctaea', 'c3dtlz4', 'case2', 'preDA.x'))
    DA = Population.new(X=da_x)
    evaluator.eval(problem, DA)

    off_x = np.loadtxt(load_to_test_resource('ctaea', 'c3dtlz4', 'case2', 'offspring.x'))
    off = Population.new(X=off_x)
    evaluator.eval(problem, off)

    post_ca_x = np.loadtxt(load_to_test_resource('ctaea', 'c3dtlz4', 'case2', 'postCA.x'))
    true_pCA = Population.new(X=post_ca_x)
    evaluator.eval(problem, true_pCA)

    post_da_x = np.loadtxt(load_to_test_resource('ctaea', 'c3dtlz4', 'case2', 'postDA.x'))
    true_pDA = Population.new(X=post_da_x)
    evaluator.eval(problem, true_pDA)

    survival = CADASurvival(ref_dirs)
    mixed = Population.merge(CA, off)
    survival.ideal_point = np.array([0., 0., 0.])

    pCA, pDA = survival.do(problem, mixed, DA, len(ref_dirs))

    pCA_X = set([tuple(x) for x in pCA.get("X")])
    tpCA_X = set([tuple(x) for x in true_pCA.get("X")])

    pDA_X = set([tuple(x) for x in pDA.get("X")])
    tpDA_X = set([tuple(x) for x in true_pDA.get("X")])

    assert pCA_X == tpCA_X
    assert pDA_X == tpDA_X


def test_mating_comparison(ref_dirs, evaluator):
    pass


def test_restricted_mating_selection(ref_dirs, evaluator):
    np.random.seed(200)
    selection = RestrictedMating(func_comp=comp_by_cv_dom_then_random)

    problem = C3DTLZ4(n_var=12, n_obj=3)
    ca_x = np.loadtxt(load_to_test_resource('ctaea', 'c3dtlz4', 'case2', 'preCA.x'))
    CA = Population.new(X=ca_x)
    evaluator.eval(problem, CA)

    da_x = np.loadtxt(load_to_test_resource('ctaea', 'c3dtlz4', 'case2', 'preDA.x'))
    DA = Population.new(X=da_x)
    evaluator.eval(problem, DA)

    Hm = Population.merge(CA, DA)
    n_pop = len(CA)

    _, rank = NonDominatedSorting().do(Hm.get('F'), return_rank=True)

    Pc = (rank[:n_pop] == 0).sum() / len(Hm)
    Pd = (rank[n_pop:] == 0).sum() / len(Hm)

    P = selection.do(None, Hm, len(CA), 2, to_pop=False)

    assert P.shape == (91, 2)
    if Pc > Pd:
        assert (P[:, 0] < n_pop).all()
    else:
        assert (P[:, 0] >= n_pop).all()
    assert (P[:, 1] >= n_pop).any()
    assert (P[:, 1] < n_pop).any()
