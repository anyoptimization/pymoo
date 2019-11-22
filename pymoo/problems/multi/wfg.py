from functools import reduce
from operator import mul

import numpy as np

from pymoo.model.problem import Problem
from pymoo.util.misc import powerset


class WFG(Problem):

    def __init__(self, n_var, n_obj, k=None, **kwargs):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=0,
                         xl=0,
                         xu=2 * np.arange(1, n_var + 1),
                         type_var=np.double,
                         **kwargs)

        if k:
            self.k = k  # number of position-related parameters
        else:
            if n_obj == 2:
                self.k = 4
            else:
                self.k = 2 * (n_obj - 1)

        self.l = n_var - self.k  # number of distance-related parameters

        self.validate_wfg_parameters(self.l, self.k, self.n_obj)

        self.S = range(2, 2 * n_obj + 1, 2)  # scaling constants vector
        self.A = [1.0] * (n_obj - 1)  # degeneracy constants vector

    def validate_wfg_parameters(self, l, k, n_obj):
        if n_obj < 2:
            raise ValueError('WFG problems must have two or more objectives.')
        if not k % (n_obj - 1) == 0:
            raise ValueError('Position parameter (k) must be divisible by number of objectives minus one.')
        if k < 4:
            raise ValueError('Position parameter (k) must be greater or equal than 4.')
        if (k + l) < n_obj:
            raise ValueError('Sum of distance and position parameters must be greater than num. of objs. (k + l >= M).')

    def destep(self, vec):
        'Removes the [2, 4, 6,...] steps.'
        return np.divide(vec, [2.0 * (i + 1) for i in range(vec.shape[1])])
        # return vec/[2.0 * (i + 1) for i in range(vec.shape[1])]

    def step_up(self, vec):
        'Introduces the [2, 4, 6,...] steps.'
        return np.multiply(vec, list([2 * (i + 1) for i in range(len(vec))]))

    def get_bounds(self):
        return [0.0] * self.n_var, [2.0 * (i + 1) for i in range(self.n_var)]

    def estimate_vec_x(self, t, a):
        x = []
        ones = np.ones(t.shape)
        for i in range(t.shape[1] - 1):
            x.append(np.max(np.row_stack([t[:, -1], ones[:, i]]), axis=0) * (t[:, i] - 0.5) + 0.5)
        # x = [max(t[-1], a[i]) * (t[i] - 0.5) + 0.5 for i in range(len(t) - 1)]
        x.append(t[:, -1])
        x = np.vstack(x).T
        return x

    def calculate_objectives(self, x, s, h):
        return np.vstack([x[:, -1] + s[i] * h[:, i] for i in range(len(s))]).T

    def _rand_optimal_position(self, n):
        return np.random.random((n, self.k))

    def _positional_to_optimal(self, K):
        suffix = np.full((len(K), self.l), 0.35)
        X = np.column_stack([K, suffix])
        return X * (2 * (np.arange(self.n_var) + 1))

    def _calc_pareto_set(self, n_pareto_points=500, *args, **kwargs):
        ps = np.ones((2 ** self.k, self.k))
        for i, s in enumerate(powerset(np.arange(self.k))):
            ps[i, s] = 0

        rnd = self._rand_optimal_position(n_pareto_points - len(ps))
        ps = np.row_stack([ps, rnd])
        ps = self._positional_to_optimal(ps)
        return ps

    def _calc_pareto_front(self, *args, n_pareto_points=500, **kwargs):
        ps = self.pareto_set(n_pareto_points=n_pareto_points)
        return self.evaluate(ps, return_values_of=["F"])


class WFG1(WFG):

    def _evaluate(self, x, out, *args, **kwargs):
        # x = np.array([[1.142805189379827, 1.7155562187004585, 3.4685478068068223, 1.6487858571160139, 8.1332125135732, 9.883066470401346, 9.148615474616461, 2.5636729043011144, 9.372048473518642, 6.555456232441863, 5.499926887100807, 22.86760581950188, 25.910481806025835, 1.2475787086121248, 25.80483111858873, 19.30209955098192, 12.974603521250007, 10.210255844641745, 25.648664191128326, 18.273246042332225, 28.806182389932978, 29.12123808230345, 6.116994656761789, 36.856215069311546],
        #               [1.142805189379827, 1.7155562187004585, 3.4685478068068223, 1.6487858571160139, 8.1332125135732, 9.883066470401346, 9.148615474616461, 2.5636729043011144, 9.372048473518642, 6.555456232441863, 5.499926887100807, 22.86760581950188, 25.910481806025835, 1.2475787086121248, 25.80483111858873, 19.30209955098192, 12.974603521250007, 10.210255844641745, 25.648664191128326, 18.273246042332225, 28.806182389932978, 29.12123808230345, 6.116994656761789, 36.856215069311546]])
        ind = self.destep(x)

        ind[:, list(range(self.k, self.n_var))] = _transformation_shift_linear(ind[:, list(range(self.k, self.n_var))],
                                                                               0.35)
        # for i in range(self.k, self.n_var):
        #     test0[i] = _transformation_shift_linear(ind[i], 0.35)

        ind[:, list(range(self.k, self.n_var))] = _transformation_bias_flat(ind[:, list(range(self.k, self.n_var))],
                                                                            0.8, 0.75, 0.85)
        # for i in range(self.k, self.n_var):
        #     ind[i] = _transformation_bias_flat(ind[i], 0.8, 0.75, 0.85)

        ind[:, list(range(self.n_var))] = np.nan_to_num(
            _transformation_bias_poly(ind[:, list(range(self.n_var))], 0.02))
        # for i in range(self.n_var):
        #     ind[i] = np.nan_to_num(_transformation_bias_poly(ind[i], 0.02))

        # set of last transition values
        w = range(2, 2 * self.n_var + 1, 2)
        gap = self.k // (self.n_obj - 1)
        t = []
        for m in range(1, self.n_obj):
            _y = ind[:, (m - 1) * gap: (m * gap)]
            _w = w[(m - 1) * gap: (m * gap)]
            t.append(_reduction_weighted_sum(_y, _w))
        # t = [_reduction_weighted_sum(ind[(m - 1) * gap : (m * gap)], w[(m - 1) * gap: (m * gap)]) for m in range(1, self.n_obj)]
        t.append(_reduction_weighted_sum(ind[:, self.k:self.n_var], w[self.k:self.n_var]))
        t = np.vstack(t).T
        _x = self.estimate_vec_x(t, self.A)

        # computation of shape vector
        h = []
        for m in range(self.n_obj - 1):
            h.append(_shape_convex(_x[:, :-1], m + 1))
        # h = [_shape_convex(x[:-1], m + 1) for m in range(self.num_obj - 1)]
        h.append(_shape_mixed(_x[:, 0]))
        h = np.column_stack(h)
        out["F"] = self.calculate_objectives(_x, self.S, h)
        # out["F"] = np.random.random((len(x), self.n_obj))

    def _rand_optimal_position(self, n):
        return np.power(np.random.random((n, self.k)), 50.0)


class WFG2(WFG):

    def _evaluate(self, x, out, *args, **kwargs):

        ind = self.destep(x)

        ind_non_sep = self.k + self.l // 2
        ind_r_sum = ind_non_sep

        ind[:, list(range(self.k, self.n_var))] = _transformation_shift_linear(ind[:, list(range(self.k, self.n_var))],
                                                                               0.35)
        # for i in range(self.k, self.num_vars):
        #     ind[i] = _transformation_shift_linear(ind[i], 0.35)

        _i = list(range(self.k, ind_non_sep - 1))
        head = lambda i: (2 * i) - self.k
        tail = lambda i: head(i) + 1

        if _i:
            ind[:, _i] = _reduction_non_sep((ind[list(map(head, _i))], ind[map(tail, _i)]), 2)
        # for i in range(self.k, ind_non_sep - 1):
        #     head = (2 * i) - self.k
        #     tail = head + 1
        #     ind[i] = _reduction_non_sep((ind[head], ind[tail]), 2)

        gap = self.k // (self.n_obj - 1)
        t = []
        for m in range(1, self.n_obj):
            _y = ind[:, (m - 1) * gap: (m * gap)]
            _w = [1.0] * gap
            t.append(_reduction_weighted_sum(_y, _w))
        t.append(_reduction_weighted_sum(ind[:, self.k:ind_r_sum], [1.0] * (ind_r_sum - self.k)))
        # t = [_reduction_weighted_sum(ind[(m - 1) * gap : (m * gap)], [1.0] * gap) for m in range(1, self.num_objs)]
        # t.append(_reduction_weighted_sum(ind[self.k:ind_r_sum], [1.0] * (ind_r_sum - self.k)))
        t = np.vstack(t).T

        _x = self.estimate_vec_x(t, self.A)
        # computation of shape vector
        h = []
        for m in range(self.n_obj - 1):
            h.append(_shape_convex(_x[:, :-1], m + 1))
        h.append(_shape_disconnected(_x[:, 0]))
        # h = [_shape_convex(x[:-1], m + 1) for m in range(self.num_objs - 1)]
        # h.append(_shape_disconnected(x[0]))
        h = np.column_stack(h)
        out["F"] = self.calculate_objectives(_x, self.S, h)

    def validate_wfg_parameters(self, l, k, n_obj):
        super().validate_wfg_parameters(l, k, n_obj)
        if not l % 2 == 0:
            raise ValueError('In WFG2/WFG3 the distance-related parameter (l) must be divisible by 2.')
        return True


class WFG3(WFG):

    def _evaluate(self, x, out, *args, **kwargs):
        # ind = self.destep(individual)
        ind = self.destep(x)

        wfg3_a = [1.0] * (self.n_obj - 1)

        if self.n_obj > 2:
            wfg3_a[1:] = [0.0] * (self.n_obj - 2)

        ind_non_sep = self.k + self.l // 2
        ind_r_sum = ind_non_sep

        _i = list(range(self.k, self.n_var))
        ind[:, _i] = _transformation_shift_linear(ind[:, _i], 0.35)
        # for i in range(self.k, self.n_var):
        #     ind[i] = _transformation_shift_linear(ind[i], 0.35)

        _i = list(range(self.k, ind_non_sep - 1))
        head = lambda i: (2 * i) - self.k
        tail = lambda i: head(i) + 1
        if _i:
            ind[:, _i] = _reduction_non_sep((ind[list(map(head, _i))], ind[list(map(tail, _i))]), 2)
        # for i in range(self.k, ind_non_sep - 1):
        #     head = (2 * i) - self.k
        #     tail = head + 1
        #     ind[i] = _reduction_non_sep((ind[head], ind[tail]), 2)

        # set of last transition values
        gap = self.k // (self.n_obj - 1)
        t = []
        for m in range(1, self.n_obj):
            _y = ind[:, (m - 1) * gap: (m * gap)]
            _w = [1.0] * gap
            t.append(_reduction_weighted_sum(_y, _w))
        t.append(_reduction_weighted_sum(ind[:, self.k:ind_r_sum], [1.0] * (ind_r_sum - self.k)))
        t = np.vstack(t).T
        # t = [_reduction_weighted_sum(ind[(m - 1) * gap : (m * gap)], [1.0] * gap) for m in range(1, self.num_objs)]
        # t.append(_reduction_weighted_sum(ind[self.k:ind_r_sum], [1.0] * (ind_r_sum - self.k)))

        _x = self.estimate_vec_x(t, wfg3_a)
        # x = self.estimate_vec_x(t, wfg3_a)

        h = []
        for m in range(self.n_obj):
            h.append(_shape_linear(_x[:, :-1], m + 1))
        h = np.column_stack(h)
        # computation of shape vector
        # h = [_shape_linear(x[:-1], m + 1) for m in range(self.n_obj)]

        out["F"] = self.calculate_objectives(_x, self.S, h)
        # return self.calculate_objectives(x, self.S, h)

    def validate_wfg_parameters(self, l, k, n_obj):
        super().validate_wfg_parameters(l, k, n_obj)
        if not l % 2 == 0:
            raise ValueError('In WFG2/WFG3 the distance-related parameter (l) must be divisible by 2.')
        return True


class WFG4(WFG):

    def _evaluate(self, x, out, *args, **kwargs):
        # ind = self.destep(individual)
        ind = self.destep(x)

        # ind = [_transformation_shift_multi_modal(item, 30.0, 10.0, 0.35) for item in ind]#5.0, 10.0, 0.35
        ind = _transformation_shift_multi_modal(ind, 30.0, 10.0, 0.35)

        # set of last transition values
        gap = self.k // (self.n_obj - 1)

        # t = [_reduction_weighted_sum(ind[(m - 1) * gap : (m * gap)], [1.0] * gap) for m in range(1, self.n_obj)]
        # t.append(_reduction_weighted_sum(ind[self.k:], [1.0] * (self.num_vars - self.k)))
        t = []
        for m in range(1, self.n_obj):
            _y = ind[:, (m - 1) * gap: (m * gap)]
            _w = [1.0] * gap
            t.append(_reduction_weighted_sum(_y, _w))
        t.append(_reduction_weighted_sum(ind[:, self.k:], [1.0] * (self.n_var - self.k)))
        t = np.vstack(t).T

        # x = self.estimate_vec_x(t, self.A)
        _x = self.estimate_vec_x(t, self.A)

        # computation of shape vector
        # h = [_shape_concave(x[:-1], m + 1) for m in range(self.num_objs)]
        h = []
        for m in range(self.n_obj):
            h.append(_shape_concave(_x[:, :-1], m + 1))
        h = np.column_stack(h)

        # return self.calculate_objectives(x, self.S, h)
        out["F"] = self.calculate_objectives(_x, self.S, h)


class WFG5(WFG):

    def _evaluate(self, x, out, *args, **kwargs):
        ind = self.destep(x)
        ind = _transformation_param_deceptive(ind)

        # set of last transition values
        gap = self.k // (self.n_obj - 1)
        t = [_reduction_weighted_sum(ind[:, (m - 1) * gap : (m * gap)], [1.0] * gap) for m in range(1, self.n_obj)]
        t.append(_reduction_weighted_sum(ind[:, self.k:], [1.0] * (self.n_var - self.k)))
        t = np.vstack(t).T

        _x = self.estimate_vec_x(t, self.A)

        # computation of shape vector
        h = [_shape_concave(_x[:, :-1], m + 1) for m in range(self.n_obj)]
        h = np.column_stack(h)

        # return self.calculate_objectives(x, self.S, h)
        out["F"] = self.calculate_objectives(_x, self.S, h)


class WFG6(WFG):

    def _evaluate(self, x, out, *args, **kwargs):
        ind = self.destep(x)

        for i in range(self.k, self.n_var):
            ind[:, i] = _transformation_shift_linear(ind[:, i], 0.35)

        # set of last transition values
        gap = self.k // (self.n_obj - 1)
        t = [_reduction_non_sep(ind[:, (m - 1) * gap : (m * gap)].T, gap) for m in range(1, self.n_obj)]
        t.append(_reduction_non_sep(ind[:, self.k:].T, self.l))
        t = np.vstack(t).T

        _x = self.estimate_vec_x(t, self.A)

        # computation of shape vector
        h = [_shape_concave(_x[:, :-1], m + 1) for m in range(self.n_obj)]
        h = np.column_stack(h)

        out["F"] = self.calculate_objectives(_x, self.S, h)


class WFG7(WFG):

    def _evaluate(self, x, out, *args, **kwargs):

        ind = self.destep(x)
        copy_ind = np.copy(ind)
        ones = [1.0] * self.n_var

        for i in range(self.k):
            aux = _reduction_weighted_sum(copy_ind[:, i + 1:], ones[i + 1:])
            ind[:, i] = _transformation_param_dependent(ind[:, i], aux)

        for i in range(self.k, self.n_var):
            ind[:, i] = _transformation_shift_linear(ind[:, i], 0.35)

        # set of last transition values
        gap = self.k // (self.n_obj - 1)

        t = []
        for m in range(1, self.n_obj):
            _y = ind[:, (m - 1) * gap: (m * gap)]
            _w = [1.0] * gap
            t.append(_reduction_weighted_sum(_y, _w))
        t.append(_reduction_weighted_sum(ind[:, self.k:], [1.0] * (self.n_var - self.k)))
        t = np.vstack(t).T

        x = self.estimate_vec_x(t, self.A)

        # computation of shape vector
        h = []
        for m in range(self.n_obj):
            h.append(_shape_concave(x[:, :-1], m + 1))
        h = np.column_stack(h)

        out["F"] = self.calculate_objectives(x, self.S, h)


class WFG8(WFG):

    def _evaluate(self, x, out, *args, **kwargs):

        ind = self.destep(x)
        copy_ind = np.copy(ind)
        ones = [1.0] * self.n_var

        for i in range(self.k, self.n_var):
            aux = _reduction_weighted_sum(copy_ind[:, :i], ones[:i])
            ind[:, i] = _transformation_param_dependent(ind[:, i], aux)

        for i in range(self.k, self.n_var):
            ind[:, i] = _transformation_shift_linear(ind[:, i], 0.35)

        # set of last transition values
        gap = self.k // (self.n_obj - 1)

        t = []
        for m in range(1, self.n_obj):
            _y = ind[:, (m - 1) * gap: (m * gap)]
            _w = [1.0] * gap
            t.append(_reduction_weighted_sum(_y, _w))
        t.append(_reduction_weighted_sum(ind[:, self.k:], [1.0] * (self.n_var - self.k)))
        t = np.vstack(t).T

        x = self.estimate_vec_x(t, self.A)

        h = []
        for m in range(self.n_obj):
            h.append(_shape_concave(x[:, :-1], m + 1))
        h = np.column_stack(h)

        out["F"] = self.calculate_objectives(x, self.S, h)

    def _positional_to_optimal(self, K):
        k, l = self.k, self.l

        for i in range(k, k + l):
            u = K.sum(axis=1) / K.shape[1]
            tmp1 = np.abs(np.floor(0.5 - u) + 0.98 / 49.98)
            tmp2 = 0.02 + 49.98 * (0.98 / 49.98 - (1.0 - 2.0 * u) * tmp1)
            suffix = np.power(0.35, np.power(tmp2, -1.0))

            K = np.column_stack([K, suffix[:, None]])

        ret = K * (2 * (np.arange(self.n_var) + 1))
        return ret


class WFG9(WFG):

    def _evaluate(self, x, out, *args, **kwargs):
        ind = self.destep(x)
        copy_ind = np.copy(ind)

        for i in range(0, self.n_var - 1):
            aux = _reduction_weighted_sum(copy_ind[:, i + 1:], [1.0] * (self.n_var - i - 1))
            ind[:, i] = _transformation_param_dependent(ind[:, i], aux)

        a = [_transformation_shift_deceptive(ind[:, i], 0.35, 0.001, 0.05) for i in range(self.k)]
        b = [_transformation_shift_multi_modal(ind[:, i], 30.0, 95.0, 0.35) for i in range(self.k, self.n_var)]
        ind = np.array(a + b).T

        # set of last transition values
        gap = self.k // (self.n_obj - 1)

        t = []
        for m in range(1, self.n_obj):
            _y = ind[:, (m - 1) * gap: (m * gap)]
            t.append(_reduction_non_sep(_y.T, gap))
        t.append(_reduction_non_sep(ind[:, self.k:].T, self.l))
        t = np.vstack(t).T

        x = self.estimate_vec_x(t, self.A)

        h = []
        for m in range(self.n_obj):
            h.append(_shape_concave(x[:, :-1], m + 1))
        h = np.column_stack(h)

        out["F"] = self.calculate_objectives(x, self.S, h)

    def _positional_to_optimal(self, K):
        k, l = self.k, self.l

        suffix = np.full((len(K), self.l), 0.0)
        X = np.column_stack([K, suffix])
        X[:, self.k + self.l - 1] = 0.35

        for i in range(self.k + self.l - 2, self.k - 1, -1):
            m = X[:, i + 1:k + l]
            val = m.sum(axis=1) / m.shape[1]
            X[:, i] = 0.35 ** ((0.02 + 1.96 * val) ** -1)

        ret = X * (2 * (np.arange(self.n_var) + 1))
        return ret


def _transformation_shift_linear(value, shift=0.35):
    'Linear shift transformation.'
    return np.fabs(value - shift) / np.fabs(np.floor(shift - value) + shift)


def _transformation_shift_deceptive(y, A=0.35, B=0.005, C=0.05):
    'Shift: Parameter Deceptive Transformation.'
    tmp1 = np.floor(y - A + B) * (1.0 - C + (A - B) / B) / (A - B)
    tmp2 = np.floor(A + B - y) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B)
    return 1.0 + (np.fabs(y - A) - B) * (tmp1 + tmp2 + 1.0 / B)


def _transformation_shift_multi_modal(y, A, B, C):
    'Shift: Parameter Multi-Modal Transformation.'
    tmp1 = np.fabs(y - C) / (2.0 * (np.floor(C - y) + C))
    tmp2 = (4.0 * A + 2.0) * np.pi * (0.5 - tmp1)
    return (1.0 + np.cos(tmp2) + 4.0 * B * np.power(tmp1, 2.0)) / (B + 2.0)


def _transformation_bias_flat(value, a, b, c):
    'Flat bias region transformation.'
    zeros = np.zeros(value.shape)
    tmp1 = np.min(np.row_stack([zeros, np.floor(value - b)]), axis=0) * (a * (b - value) / b)
    # tmp1 = min(0.0, np.floor(value - b))* (a * (b - value) / b)
    tmp2 = np.min(np.row_stack([zeros, np.floor(c - value)]), axis=0) * ((1.0 - a) * (value - c) / (1.0 - c))
    # tmp2 = min(0.0, np.floor(c - value)) * ((1.0 - a) * (value - c) / (1.0 - c))
    return a + tmp1 - tmp2


def _transformation_bias_poly(y, alpha):
    'Polynomial bias transformation.'
    aux = np.power(y, alpha)
    # aux = y ** alpha
    return aux


def _transformation_param_dependent(y, y_deg, A=0.98 / 49.98, B=0.02, C=50.0):
    'Parameter dependent bias transformation.'
    aux = A - (1.0 - 2.0 * y_deg) * np.fabs(np.floor(0.5 - y_deg) + A)
    return pow(y, B + (C - B) * aux)


def _transformation_param_deceptive(y, A=0.35, B=0.001, C=0.05):
    'Shift: Parameter Deceptive Transformation.'
    tmp1 = np.floor(y - A + B) * (1.0 - C + (A - B) / B) / (A - B)
    tmp2 = np.floor(A + B - y) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B)
    return 1.0 + (np.fabs(y - A) - B) * (tmp1 + tmp2 + 1.0 / B)


def _reduction_weighted_sum(y, w):
    'Weighted sum reduction transformation.'
    return np.dot(y, w) / sum(w)


def _reduction_non_sep(y, A):
    'Non-Separable reduction transformation.'
    numerator = 0.0
    for j in range(len(y)):
        numerator += y[j]
        for k in range(A - 1):  # To verify the constant (1 or 2)
            numerator += np.fabs(y[j] - y[(1 + j + k) % len(y)])
    tmp = np.ceil(A / 2.0)
    denominator = len(y) * tmp * (1.0 + 2.0 * A - 2 * tmp) / A
    return numerator / denominator


def _shape_convex(x, m):
    'Convex Pareto front shape function.'
    if m == 1:
        result = np.array(list(1.0 - np.cos(0.5 * xi * np.pi) for xi in x[:, :x.shape[1]])) * 1.0
        # result = reduce(mul, (1.0 - np.cos(0.5 * xi[:] * np.pi) for xi in x[:x.shape[1]]), 1.0)
    elif 1 < m <= len(x):
        result = reduce(mul, (1.0 - np.cos(0.5 * xi * np.pi) for xi in x[:len(x) - m + 1]), 1.0)
        result *= 1.0 - np.sin(0.5 * x[len(x) - m + 1] * np.pi)
    else:
        result = 1.0 - np.sin(0.5 * x[0] * np.pi)
    return result


def _shape_mixed(x, A=5.0, alpha=1.0):
    'Convex/concave mixed Pareto front shape function.'
    aux = 2.0 * A * np.pi
    return np.array(pow(1.0 - x - (np.cos(aux * x + 0.5 * np.pi) / aux), alpha))[:, None]


def _shape_disconnected(x, alpha=1.0, beta=1.0, A=5.0):
    'Disconnected Pareto front shape function.'
    aux = np.cos(A * np.pi * pow(x, beta))
    return 1.0 - pow(x, alpha) * pow(aux, 2)


def _shape_linear(x, m):
    'Linear Pareto optimal front shape function.'
    if m == 1:
        result = np.array(list(xi for xi in x[:, :x.shape[1]])) * 1.0
        # result = reduce(mul, (xi for xi in x[:len(x)]), 1.0)
    elif 1 < m <= x.shape[1]:
        # elif 1 < m <= len(x):
        result = np.array(list(xi for xi in x[:, :x.shape[1] - m + 1])) * 1.0
        result *= (1.0 - x[x.shape[1] - m + 1])
        # result = reduce(mul, (xi for xi in x[:len(x) - m + 1]), 1.0)
        # result *= (1.0 - x[len(x) - m + 1])
    else:
        result = 1.0 - x[:, 0]
        # result = 1.0 - x[0]
    return result


def _shape_concave(x, m):
    'Concave Pareto optimal shape function.'
    if m == 1:
        # _result = reduce(mul, (np.sin(0.5 * xi * np.pi) for xi in x[:len(x)]), 1.0)
        result = np.prod(np.sin(0.5 * x[:, :x.shape[1]] * np.pi), axis=1)

    elif 1 < m <= len(x):
        # result = reduce(mul, (np.sin(0.5 * xi * np.pi) for xi in x[:len(x) - m + 1]), 1.0)
        result = np.prod(np.sin(0.5 * x[:, :x.shape[1] - m + 1] * np.pi), axis=1)
        result *= np.cos(0.5 * x[:, x.shape[1] - m + 1] * np.pi)
    else:
        result = np.cos(0.5 * x[:, 0] * np.pi)
    return result
