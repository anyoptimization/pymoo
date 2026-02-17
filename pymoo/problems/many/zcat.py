from __future__ import annotations

import math

import numpy as np

from pymoo.core.problem import Problem
from pymoo.util.remote import Remote

"""
.. module:: ZCAT
   :platform: Unix, Windows
   :synopsis: ZCAT problem family of multi-objective problems.
"""

_DBL_EPSILON = 2.220446049250313e-16
_DOUBLE_MIN_VALUE = 5e-324
_PI = math.pi
_E = math.e

_COMPLICATED_G_FUNCTION_BY_PROBLEM = {
    1: 4,
    2: 5,
    3: 2,
    4: 7,
    5: 9,
    6: 4,
    7: 5,
    8: 2,
    9: 7,
    10: 9,
    11: 3,
    12: 10,
    13: 1,
    14: 6,
    15: 8,
    16: 10,
    17: 1,
    18: 8,
    19: 6,
    20: 3,
}

_ONE_DIMENSIONAL_PARETO_SET_PROBLEMS = (14, 15, 16)


def _fix_to_01(value: float) -> float:
    if value <= 0.0 and value >= -_DBL_EPSILON:
        return 0.0
    if value >= 1.0 and value <= 1.0 + _DBL_EPSILON:
        return 1.0
    return value


def _leq(value: float, bound: float) -> bool:
    return value < bound or abs(bound - value) < _DBL_EPSILON


def _value_in(value: float, lower: float, upper: float) -> bool:
    return _leq(lower, value) and _leq(value, upper)


def _all_values_in(values: np.ndarray, m: int, lower: float, upper: float) -> bool:
    limit = min(max(m, 0), int(values.shape[0]))
    for i in range(limit):
        if not _value_in(float(values[i]), lower, upper):
            return False
    return True


def _theta_j(j: int, m: int, n: int) -> float:
    return 2.0 * _PI * (j - 1.0) / (n - m)


def _g0(y: np.ndarray, m: int, n: int) -> np.ndarray:
    size = n - m
    g = np.empty(size, dtype=float)
    g.fill(0.2210)
    return g


def _g1(y: np.ndarray, m: int, n: int) -> np.ndarray:
    size = n - m
    g = np.empty(size, dtype=float)
    for j in range(1, size + 1):
        total = 0.0
        angle = _theta_j(j, m, n)
        for i in range(1, m + 1):
            total += math.sin(1.5 * _PI * float(y[i - 1]) + angle)
        g[j - 1] = total / (2.0 * m) + 0.5
    return g


def _g2(y: np.ndarray, m: int, n: int) -> np.ndarray:
    size = n - m
    g = np.empty(size, dtype=float)
    for j in range(1, size + 1):
        total = 0.0
        angle = _theta_j(j, m, n)
        for i in range(1, m + 1):
            yi = float(y[i - 1])
            total += (yi**2.0) * math.sin(4.5 * _PI * yi + angle)
        g[j - 1] = total / (2.0 * m) + 0.5
    return g


def _g3(y: np.ndarray, m: int, n: int) -> np.ndarray:
    size = n - m
    g = np.empty(size, dtype=float)
    for j in range(1, size + 1):
        total = 0.0
        angle = _theta_j(j, m, n)
        for i in range(1, m + 1):
            total += math.cos(_PI * float(y[i - 1]) + angle) ** 2.0
        g[j - 1] = total / m
    return g


def _g4(y: np.ndarray, m: int, n: int) -> np.ndarray:
    size = n - m
    g = np.empty(size, dtype=float)
    mu = float(np.sum(y[:m])) / m
    for j in range(1, size + 1):
        g[j - 1] = (mu / 2.0) * math.cos(4.0 * _PI * mu + _theta_j(j, m, n)) + 0.5
    return g


def _g5(y: np.ndarray, m: int, n: int) -> np.ndarray:
    size = n - m
    g = np.empty(size, dtype=float)
    for j in range(1, size + 1):
        total = 0.0
        angle = _theta_j(j, m, n)
        for i in range(1, m + 1):
            total += math.sin(2.0 * _PI * float(y[i - 1]) - 1.0 + angle) ** 3.0
        g[j - 1] = total / (2.0 * m) + 0.5
    return g


def _g6(y: np.ndarray, m: int, n: int) -> np.ndarray:
    size = n - m
    g = np.empty(size, dtype=float)
    denominator = -10.0 * math.exp(-2.0 / 5.0) - math.exp(-1.0) + 10.0 + _E
    for j in range(1, size + 1):
        s1 = 0.0
        s2 = 0.0
        angle = _theta_j(j, m, n)
        for i in range(1, m + 1):
            yi = float(y[i - 1])
            s1 += yi**2.0
            s2 += math.cos(11.0 * _PI * yi + angle) ** 3.0
        s1 /= m
        s2 /= m
        numerator = -10.0 * math.exp((-2.0 / 5.0) * math.sqrt(s1)) - math.exp(s2) + 10.0 + _E
        g[j - 1] = numerator / denominator
    return g


def _g7(y: np.ndarray, m: int, n: int) -> np.ndarray:
    size = n - m
    g = np.empty(size, dtype=float)
    mu = float(np.sum(y[:m])) / m
    denominator = 1.0 + _E - math.exp(-1.0)
    for j in range(1, size + 1):
        angle = _theta_j(j, m, n)
        g[j - 1] = (mu + math.exp(math.sin(7.0 * _PI * mu - _PI / 2.0 + angle)) - math.exp(-1.0)) / denominator
    return g


def _g8(y: np.ndarray, m: int, n: int) -> np.ndarray:
    size = n - m
    g = np.empty(size, dtype=float)
    for j in range(1, size + 1):
        total = 0.0
        angle = _theta_j(j, m, n)
        for i in range(1, m + 1):
            total += abs(math.sin(2.5 * _PI * (float(y[i - 1]) - 0.5) + angle))
        g[j - 1] = total / m
    return g


def _g9(y: np.ndarray, m: int, n: int) -> np.ndarray:
    size = n - m
    g = np.empty(size, dtype=float)
    mu = float(np.sum(y[:m])) / m
    for j in range(1, size + 1):
        total = 0.0
        angle = _theta_j(j, m, n)
        for i in range(1, m + 1):
            total += abs(math.sin(2.5 * _PI * float(y[i - 1]) - _PI / 2.0 + angle))
        g[j - 1] = mu / 2.0 - total / (2.0 * m) + 0.5
    return g


def _g10(y: np.ndarray, m: int, n: int) -> np.ndarray:
    size = n - m
    g = np.empty(size, dtype=float)
    denominator = 2.0 * (m**3.0)
    for j in range(1, size + 1):
        total = 0.0
        angle = _theta_j(j, m, n)
        for i in range(1, m + 1):
            total += math.sin((4.0 * float(y[i - 1]) - 2.0) * _PI + angle)
        g[j - 1] = (total**3.0) / denominator + 0.5
    return g


def _evaluate_g(g_function_id: int, y: np.ndarray, m: int, n: int) -> np.ndarray:
    if g_function_id == 0:
        return _g0(y, m, n)
    if g_function_id == 1:
        return _g1(y, m, n)
    if g_function_id == 2:
        return _g2(y, m, n)
    if g_function_id == 3:
        return _g3(y, m, n)
    if g_function_id == 4:
        return _g4(y, m, n)
    if g_function_id == 5:
        return _g5(y, m, n)
    if g_function_id == 6:
        return _g6(y, m, n)
    if g_function_id == 7:
        return _g7(y, m, n)
    if g_function_id == 8:
        return _g8(y, m, n)
    if g_function_id == 9:
        return _g9(y, m, n)
    if g_function_id == 10:
        return _g10(y, m, n)
    raise ValueError(f"Unsupported g-function id: {g_function_id}")


def _z1(j_values: np.ndarray) -> float:
    j_size = int(j_values.shape[0])
    return (10.0 / j_size) * float(np.sum(j_values * j_values))


def _z2(j_values: np.ndarray) -> float:
    return 10.0 * float(np.max(np.abs(j_values)))


def _z3(j_values: np.ndarray) -> float:
    j_size = int(j_values.shape[0])
    k = 5.0
    total = 0.0
    for value in j_values:
        total += (float(value) ** 2.0 - math.cos((2.0 * k - 1.0) * _PI * float(value)) + 1.0) / 3.0
    return (10.0 / j_size) * total


def _z4(j_values: np.ndarray) -> float:
    j_size = int(j_values.shape[0])
    k = 5.0
    pow1 = float(np.max(np.abs(j_values)))
    pow2 = 0.0
    for value in j_values:
        pow2 += 0.5 * (math.cos((2.0 * k - 1.0) * _PI * float(value)) + 1.0)
    numerator = math.exp(pow1**0.5) - math.exp(pow2 / j_size) - 1.0 + _E
    return (10.0 / (2.0 * _E - 2.0)) * numerator


def _z5(j_values: np.ndarray) -> float:
    j_size = int(j_values.shape[0])
    total = 0.0
    for value in j_values:
        total += abs(float(value)) ** 0.002
    return -0.7 * _z3(j_values) + (10.0 / j_size) * total


def _z6(j_values: np.ndarray) -> float:
    j_size = int(j_values.shape[0])
    total = 0.0
    for value in j_values:
        total += abs(float(value))
    return -0.7 * _z4(j_values) + 10.0 * (total / j_size) ** 0.002


def _zbias(z_value: float) -> float:
    return abs(z_value) ** 0.05


def _get_j(objective_index: int, number_of_objectives: int, w: np.ndarray, w_size: int) -> np.ndarray:
    values = []
    for j in range(1, w_size + 1):
        if (j - objective_index) % number_of_objectives == 0:
            values.append(float(w[j - 1]))
    if not values:
        values.append(float(w[0]))
    return np.asarray(values, dtype=float)


def _evaluate_z(j_values: np.ndarray, objective_index: int, imbalance: bool, level: int) -> float:
    if imbalance:
        return _z4(j_values) if objective_index % 2 == 0 else _z1(j_values)

    if level == 1:
        return _z1(j_values)
    if level == 2:
        return _z2(j_values)
    if level == 3:
        return _z3(j_values)
    if level == 4:
        return _z4(j_values)
    if level == 5:
        return _z5(j_values)
    if level == 6:
        return _z6(j_values)
    return _z1(j_values)


def _f1(y: np.ndarray, n_obj: int) -> np.ndarray:
    f = np.zeros(n_obj, dtype=float)

    f[0] = 1.0
    for i in range(1, n_obj):
        f[0] *= math.sin(float(y[i - 1]) * _PI / 2.0)
    f[0] = _fix_to_01(float(f[0]))

    for j in range(2, n_obj):
        value = 1.0
        for i in range(1, n_obj - j + 1):
            value *= math.sin(float(y[i - 1]) * _PI / 2.0)
        value *= math.cos(float(y[n_obj - j]) * _PI / 2.0)
        f[j - 1] = _fix_to_01(value)

    f[n_obj - 1] = _fix_to_01(1.0 - math.sin(float(y[0]) * _PI / 2.0))
    return f


def _f2(y: np.ndarray, n_obj: int) -> np.ndarray:
    f = np.zeros(n_obj, dtype=float)

    value = 1.0
    for i in range(1, n_obj):
        value *= 1.0 - math.cos(float(y[i - 1]) * _PI / 2.0)
    f[0] = value

    for j in range(2, n_obj):
        value = 1.0
        for i in range(1, n_obj - j + 1):
            value *= 1.0 - math.cos(float(y[i - 1]) * _PI / 2.0)
        value *= 1.0 - math.sin(float(y[n_obj - j]) * _PI / 2.0)
        f[j - 1] = value

    f[n_obj - 1] = 1.0 - math.sin(float(y[0]) * _PI / 2.0)
    return f


def _f3(y: np.ndarray, n_obj: int) -> np.ndarray:
    f = np.zeros(n_obj, dtype=float)

    total = 0.0
    for i in range(1, n_obj):
        total += float(y[i - 1])
    f[0] = total / (n_obj - 1.0)

    for j in range(2, n_obj):
        total = 0.0
        for i in range(1, n_obj - j + 1):
            total += float(y[i - 1])
        total += 1.0 - float(y[n_obj - j])
        f[j - 1] = total / (n_obj - j + 1.0)

    f[n_obj - 1] = 1.0 - float(y[0])
    return f


def _f4(y: np.ndarray, n_obj: int) -> np.ndarray:
    f = np.zeros(n_obj, dtype=float)

    total = 0.0
    for j in range(1, n_obj):
        f[j - 1] = float(y[j - 1])
        total += float(y[j - 1])

    f[n_obj - 1] = 1.0 - total / (n_obj - 1.0)
    return f


def _f5(y: np.ndarray, n_obj: int) -> np.ndarray:
    f = np.zeros(n_obj, dtype=float)

    total = 0.0
    for j in range(1, n_obj):
        f[j - 1] = float(y[j - 1])
        total += 1.0 - float(y[j - 1])

    numerator = math.exp(total / (n_obj - 1.0)) ** 8.0 - 1.0
    denominator = math.exp(1.0) ** 8.0 - 1.0
    f[n_obj - 1] = numerator / denominator
    return f


def _f6(y: np.ndarray, n_obj: int) -> np.ndarray:
    f = np.zeros(n_obj, dtype=float)
    k = 40.0
    r = 0.05

    mu = 0.0
    for j in range(1, n_obj):
        f[j - 1] = float(y[j - 1])
        mu += float(y[j - 1])
    mu /= n_obj - 1.0

    numerator = (1.0 + math.exp(2.0 * k * mu - k)) ** -1.0 - r * mu - (1.0 + math.exp(k)) ** -1.0 + r
    denominator = (1.0 + math.exp(-k)) ** -1.0 - (1.0 + math.exp(k)) ** -1.0 + r
    f[n_obj - 1] = numerator / denominator
    return f


def _f7(y: np.ndarray, n_obj: int) -> np.ndarray:
    f = np.zeros(n_obj, dtype=float)

    total = 0.0
    for j in range(1, n_obj):
        f[j - 1] = float(y[j - 1])
        total += (0.5 - float(y[j - 1])) ** 5.0

    f[n_obj - 1] = total / (2.0 * (n_obj - 1.0) * (0.5**5.0)) + 0.5
    return f


def _f8(y: np.ndarray, n_obj: int) -> np.ndarray:
    f = np.zeros(n_obj, dtype=float)

    value = 1.0
    for i in range(1, n_obj):
        value *= 1.0 - math.sin(float(y[i - 1]) * _PI / 2.0)
    f[0] = 1.0 - value

    for j in range(2, n_obj):
        value = 1.0
        for i in range(1, n_obj - j + 1):
            value *= 1.0 - math.sin(float(y[i - 1]) * _PI / 2.0)
        value *= 1.0 - math.cos(float(y[n_obj - j]) * _PI / 2.0)
        f[j - 1] = 1.0 - value

    f[n_obj - 1] = math.cos(float(y[0]) * _PI / 2.0)
    return f


def _f9(y: np.ndarray, n_obj: int) -> np.ndarray:
    f = np.zeros(n_obj, dtype=float)

    total = 0.0
    for i in range(1, n_obj):
        total += math.sin(float(y[i - 1]) * _PI / 2.0)
    f[0] = total / (n_obj - 1.0)

    for j in range(2, n_obj):
        total = 0.0
        for i in range(1, n_obj - j + 1):
            total += math.sin(float(y[i - 1]) * _PI / 2.0)
        total += math.cos(float(y[n_obj - j]) * _PI / 2.0)
        f[j - 1] = total / (n_obj - j + 1.0)

    f[n_obj - 1] = math.cos(float(y[0]) * _PI / 2.0)
    return f


def _f10(y: np.ndarray, n_obj: int) -> np.ndarray:
    f = np.zeros(n_obj, dtype=float)
    r = 0.02

    total = 0.0
    for j in range(1, n_obj):
        total += 1.0 - float(y[j - 1])
        f[j - 1] = float(y[j - 1])

    numerator = (r**-1.0) - ((total / (n_obj - 1.0) + r) ** -1.0)
    denominator = (r**-1.0) - ((1.0 + r) ** -1.0)
    f[n_obj - 1] = numerator / denominator
    return f


def _f11(y: np.ndarray, n_obj: int) -> np.ndarray:
    f = np.zeros(n_obj, dtype=float)
    k = 4.0

    total = 0.0
    for i in range(1, n_obj):
        total += float(y[i - 1])
    f[0] = total / (n_obj - 1.0)

    for j in range(2, n_obj):
        total = 0.0
        for i in range(1, n_obj - j + 1):
            total += float(y[i - 1])
        total += 1.0 - float(y[n_obj - j])
        f[j - 1] = total / (n_obj - j + 1.0)

    y0 = float(y[0])
    f[n_obj - 1] = (math.cos((2.0 * k - 1.0) * y0 * _PI) + 2.0 * y0 + 4.0 * k * (1.0 - y0) - 1.0) / (4.0 * k)
    return f


def _f12(y: np.ndarray, n_obj: int) -> np.ndarray:
    f = np.zeros(n_obj, dtype=float)
    k = 3.0

    value = 1.0
    for i in range(1, n_obj):
        value *= 1.0 - float(y[i - 1])
    f[0] = 1.0 - value

    for j in range(2, n_obj):
        value = 1.0
        for i in range(1, n_obj - j + 1):
            value *= 1.0 - float(y[i - 1])
        value *= float(y[n_obj - j])
        f[j - 1] = 1.0 - value

    y0 = float(y[0])
    f[n_obj - 1] = (math.cos((2.0 * k - 1.0) * y0 * _PI) + 2.0 * y0 + 4.0 * k * (1.0 - y0) - 1.0) / (4.0 * k)
    return f


def _f13(y: np.ndarray, n_obj: int) -> np.ndarray:
    f = np.zeros(n_obj, dtype=float)
    k = 3.0

    total = 0.0
    for i in range(1, n_obj):
        total += math.sin(float(y[i - 1]) * _PI / 2.0)
    f[0] = 1.0 - total / (n_obj - 1.0)

    for j in range(2, n_obj):
        total = 0.0
        for i in range(1, n_obj - j + 1):
            total += math.sin(float(y[i - 1]) * _PI / 2.0)
        total += math.cos(float(y[n_obj - j]) * _PI / 2.0)
        f[j - 1] = 1.0 - total / (n_obj - j + 1.0)

    y0 = float(y[0])
    f[n_obj - 1] = 1.0 - (
        math.cos((2.0 * k - 1.0) * y0 * _PI) + 2.0 * y0 + 4.0 * k * (1.0 - y0) - 1.0
    ) / (4.0 * k)
    return f


def _f14(y: np.ndarray, n_obj: int) -> np.ndarray:
    f = np.zeros(n_obj, dtype=float)
    y0 = float(y[0])
    sin_term = math.sin(y0 * _PI / 2.0)

    f[0] = sin_term**2.0
    for j in range(2, n_obj - 1):
        f[j - 1] = sin_term ** (2.0 + (j - 1.0) / (n_obj - 2.0))

    if n_obj > 2:
        f[n_obj - 2] = 0.5 * (1.0 + math.sin(6.0 * y0 * _PI / 2.0 - _PI / 2.0))

    f[n_obj - 1] = math.cos(y0 * _PI / 2.0)
    return f


def _f15(y: np.ndarray, n_obj: int) -> np.ndarray:
    f = np.zeros(n_obj, dtype=float)
    k = 3.0
    y0 = float(y[0])

    for j in range(1, n_obj):
        f[j - 1] = y0 ** (1.0 + (j - 1.0) / (4.0 * n_obj))

    f[n_obj - 1] = (math.cos((2.0 * k - 1.0) * y0 * _PI) + 2.0 * y0 + 4.0 * k * (1.0 - y0) - 1.0) / (4.0 * k)
    return f


def _f16(y: np.ndarray, n_obj: int) -> np.ndarray:
    f = np.zeros(n_obj, dtype=float)
    k = 5.0
    y0 = float(y[0])
    sin_term = math.sin(y0 * _PI / 2.0)

    f[0] = sin_term
    for j in range(2, n_obj - 1):
        f[j - 1] = sin_term ** (1.0 + (j - 1.0) / (n_obj - 2.0))

    if n_obj > 2:
        f[n_obj - 2] = 0.5 * (1.0 + math.sin(10.0 * y0 * _PI / 2.0 - _PI / 2.0))

    f[n_obj - 1] = (math.cos((2.0 * k - 1.0) * y0 * _PI) + 2.0 * y0 + 4.0 * k * (1.0 - y0) - 1.0) / (4.0 * k)
    return f


def _f17(y: np.ndarray, n_obj: int) -> np.ndarray:
    f = np.zeros(n_obj, dtype=float)

    wedge_flag = True
    for j in range(0, n_obj - 1):
        yj = float(y[j])
        if yj < 0.0 or yj > 0.5:
            wedge_flag = False
            break

    total = 0.0
    for j in range(1, n_obj):
        if wedge_flag:
            f[j - 1] = float(y[0])
        else:
            f[j - 1] = float(y[j - 1])
            total += 1.0 - float(y[j - 1])

    if wedge_flag:
        numerator = math.exp(1.0 - float(y[0])) ** 8.0 - 1.0
    else:
        numerator = math.exp(total / (n_obj - 1.0)) ** 8.0 - 1.0
    denominator = math.exp(1.0) ** 8.0 - 1.0
    f[n_obj - 1] = numerator / denominator
    return f


def _f18(y: np.ndarray, n_obj: int) -> np.ndarray:
    f = np.zeros(n_obj, dtype=float)

    in_first = _all_values_in(y, n_obj - 1, 0.0, 0.4)
    in_second = _all_values_in(y, n_obj - 1, 0.6, 1.0)
    wedge_flag = in_first or in_second

    total = 0.0
    for j in range(1, n_obj):
        if wedge_flag:
            f[j - 1] = float(y[0])
        else:
            f[j - 1] = float(y[j - 1])
            total += (0.5 - float(y[j - 1])) ** 5.0

    if wedge_flag:
        y0 = float(y[0])
        f[n_obj - 1] = ((0.5 - y0) ** 5.0 + 0.5**5.0) / (2.0 * (0.5**5.0))
    else:
        f[n_obj - 1] = total / (2.0 * (n_obj - 1.0) * (0.5**5.0)) + 0.5
    return f


def _f19(y: np.ndarray, n_obj: int) -> np.ndarray:
    f = np.zeros(n_obj, dtype=float)
    a = 5.0

    flag_deg = _value_in(float(y[0]), 0.0, 0.2) or _value_in(float(y[0]), 0.4, 0.6)

    mu = 0.0
    for j in range(1, n_obj):
        mu += float(y[j - 1])
        f[j - 1] = float(y[0]) if flag_deg else float(y[j - 1])
        f[j - 1] = _fix_to_01(float(f[j - 1]))
    mu = float(y[0]) if flag_deg else mu / (n_obj - 1.0)

    f[n_obj - 1] = 1.0 - mu - math.cos(2.0 * a * _PI * mu + _PI / 2.0) / (2.0 * a * _PI)
    f[n_obj - 1] = _fix_to_01(float(f[n_obj - 1]))
    return f


def _f20(y: np.ndarray, n_obj: int) -> np.ndarray:
    f = np.zeros(n_obj, dtype=float)

    deg_flag = _value_in(float(y[0]), 0.1, 0.4) or _value_in(float(y[0]), 0.6, 0.9)

    total = 0.0
    for j in range(1, n_obj):
        total += (0.5 - float(y[j - 1])) ** 5.0
        f[j - 1] = float(y[0]) if deg_flag else float(y[j - 1])

    if deg_flag:
        y0 = float(y[0])
        f[n_obj - 1] = ((0.5 - y0) ** 5.0 + 0.5**5.0) / (2.0 * (0.5**5.0))
    else:
        f[n_obj - 1] = total / (2.0 * (n_obj - 1.0) * (0.5**5.0)) + 0.5
    return f


def _evaluate_f(function_id: int, y: np.ndarray, n_obj: int) -> np.ndarray:
    if function_id == 1:
        return _f1(y, n_obj)
    if function_id == 2:
        return _f2(y, n_obj)
    if function_id == 3:
        return _f3(y, n_obj)
    if function_id == 4:
        return _f4(y, n_obj)
    if function_id == 5:
        return _f5(y, n_obj)
    if function_id == 6:
        return _f6(y, n_obj)
    if function_id == 7:
        return _f7(y, n_obj)
    if function_id == 8:
        return _f8(y, n_obj)
    if function_id == 9:
        return _f9(y, n_obj)
    if function_id == 10:
        return _f10(y, n_obj)
    if function_id == 11:
        return _f11(y, n_obj)
    if function_id == 12:
        return _f12(y, n_obj)
    if function_id == 13:
        return _f13(y, n_obj)
    if function_id == 14:
        return _f14(y, n_obj)
    if function_id == 15:
        return _f15(y, n_obj)
    if function_id == 16:
        return _f16(y, n_obj)
    if function_id == 17:
        return _f17(y, n_obj)
    if function_id == 18:
        return _f18(y, n_obj)
    if function_id == 19:
        return _f19(y, n_obj)
    if function_id == 20:
        return _f20(y, n_obj)
    raise ValueError(f"Unsupported F-function id: {function_id}")


class ZCAT(Problem):
    def __init__(
        self,
        problem_id: int,
        n_var: int = 30,
        n_obj: int = 2,
        complicated_pareto_set: bool = False,
        level: int = 1,
        bias: bool = False,
        imbalance: bool = False,
        **kwargs,
    ) -> None:
        if problem_id < 1 or problem_id > 20:
            raise ValueError("ZCAT problem id must be in [1, 20].")
        if n_obj < 2:
            raise ValueError("ZCAT requires at least two objectives.")
        if n_var <= 0:
            raise ValueError("ZCAT requires n_var > 0.")

        min_required_n_var = 1 if problem_id in _ONE_DIMENSIONAL_PARETO_SET_PROBLEMS else n_obj - 1
        if n_var < min_required_n_var:
            raise ValueError(
                f"ZCAT{problem_id} requires n_var >= {min_required_n_var} for n_obj={n_obj}. "
                f"Received n_var={n_var}."
            )

        self.problem_id = int(problem_id)
        self.complicated_pareto_set = bool(complicated_pareto_set)
        self.level = int(level)
        self.bias = bool(bias)
        self.imbalance = bool(imbalance)

        indices = np.arange(1.0, n_var + 1.0, dtype=float)
        xl = -0.5 * indices
        xu = 0.5 * indices

        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, vtype=float, **kwargs)

    def _pareto_set_dimension(self, y0: float) -> int:
        if self.problem_id in _ONE_DIMENSIONAL_PARETO_SET_PROBLEMS:
            return 1

        if self.problem_id == 19:
            if _value_in(y0, 0.0, 0.2) or _value_in(y0, 0.4, 0.6):
                return 1
            return self.n_obj - 1

        if self.problem_id == 20:
            if _value_in(y0, 0.1, 0.4) or _value_in(y0, 0.6, 0.9):
                return 1
            return self.n_obj - 1

        return self.n_obj - 1

    def _g_function_id(self) -> int:
        if not self.complicated_pareto_set:
            return 0
        return _COMPLICATED_G_FUNCTION_BY_PROBLEM[self.problem_id]

    def _evaluate_row(self, x_row: np.ndarray) -> np.ndarray:
        y = (x_row - self.xl) / (self.xu - self.xl)
        pareto_set_dimension = self._pareto_set_dimension(float(y[0]))

        alpha = _evaluate_f(self.problem_id, y, self.n_obj)
        for objective_index in range(1, self.n_obj + 1):
            alpha[objective_index - 1] = (objective_index**2.0) * float(alpha[objective_index - 1])

        beta = np.zeros(self.n_obj, dtype=float)
        if pareto_set_dimension != self.n_var:
            g_values = _evaluate_g(self._g_function_id(), y, pareto_set_dimension, self.n_var)
            z_values = y[pareto_set_dimension:] - g_values

            for idx in range(z_values.shape[0]):
                if abs(float(z_values[idx])) < _DOUBLE_MIN_VALUE:
                    z_values[idx] = 0.0

            if self.bias:
                w_values = np.empty_like(z_values)
                for idx in range(z_values.shape[0]):
                    w_values[idx] = _zbias(float(z_values[idx]))
            else:
                w_values = z_values

            w_size = self.n_var - pareto_set_dimension
            for objective_index in range(1, self.n_obj + 1):
                j_values = _get_j(objective_index, self.n_obj, w_values, w_size)
                z_score = _evaluate_z(j_values, objective_index, self.imbalance, self.level)
                beta[objective_index - 1] = (objective_index**2.0) * z_score

        return alpha + beta

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.asarray(x, dtype=float)
        f = np.empty((x.shape[0], self.n_obj), dtype=float)
        for row_index in range(x.shape[0]):
            f[row_index, :] = self._evaluate_row(x[row_index, :])
        out["F"] = f

    def _calc_pareto_front(self, n_pareto_points=None):
        if self.n_obj not in (2, 3, 4, 6):
            raise Exception("Only n_obj in {2, 3, 4, 6} has a bundled reference front for ZCAT.")

        file_candidates = []
        if self.n_obj == 2:
            file_candidates.extend(
                [
                    f"ZCAT{self.problem_id}.pf",
                    f"ZCAT{self.problem_id}.2D.pf",
                    f"zcat{self.problem_id}.pf",
                    f"zcat{self.problem_id}.2d.pf",
                ]
            )
        else:
            file_candidates.extend(
                [
                    f"ZCAT{self.problem_id}.{self.n_obj}D.pf",
                    f"zcat{self.problem_id}.{self.n_obj}d.pf",
                ]
            )

        remote = Remote.get_instance()
        pf = None
        for filename in file_candidates:
            try:
                pf = remote.load("pymoo", "pf", "ZCAT", filename)
                break
            except Exception:
                pass

        if pf is None:
            raise Exception(
                f"Reference front for ZCAT{self.problem_id} with n_obj={self.n_obj} not found in pymoo/pf/ZCAT."
            )

        if n_pareto_points is not None and n_pareto_points > 0 and len(pf) > n_pareto_points:
            idx = np.linspace(0, len(pf) - 1, n_pareto_points).astype(int)
            pf = pf[idx]

        return pf


class ZCAT1(ZCAT):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)


class ZCAT2(ZCAT):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)


class ZCAT3(ZCAT):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)


class ZCAT4(ZCAT):
    def __init__(self, *args, **kwargs):
        super().__init__(4, *args, **kwargs)


class ZCAT5(ZCAT):
    def __init__(self, *args, **kwargs):
        super().__init__(5, *args, **kwargs)


class ZCAT6(ZCAT):
    def __init__(self, *args, **kwargs):
        super().__init__(6, *args, **kwargs)


class ZCAT7(ZCAT):
    def __init__(self, *args, **kwargs):
        super().__init__(7, *args, **kwargs)


class ZCAT8(ZCAT):
    def __init__(self, *args, **kwargs):
        super().__init__(8, *args, **kwargs)


class ZCAT9(ZCAT):
    def __init__(self, *args, **kwargs):
        super().__init__(9, *args, **kwargs)


class ZCAT10(ZCAT):
    def __init__(self, *args, **kwargs):
        super().__init__(10, *args, **kwargs)


class ZCAT11(ZCAT):
    def __init__(self, *args, **kwargs):
        super().__init__(11, *args, **kwargs)


class ZCAT12(ZCAT):
    def __init__(self, *args, **kwargs):
        super().__init__(12, *args, **kwargs)


class ZCAT13(ZCAT):
    def __init__(self, *args, **kwargs):
        super().__init__(13, *args, **kwargs)


class ZCAT14(ZCAT):
    def __init__(self, *args, **kwargs):
        super().__init__(14, *args, **kwargs)


class ZCAT15(ZCAT):
    def __init__(self, *args, **kwargs):
        super().__init__(15, *args, **kwargs)


class ZCAT16(ZCAT):
    def __init__(self, *args, **kwargs):
        super().__init__(16, *args, **kwargs)


class ZCAT17(ZCAT):
    def __init__(self, *args, **kwargs):
        super().__init__(17, *args, **kwargs)


class ZCAT18(ZCAT):
    def __init__(self, *args, **kwargs):
        super().__init__(18, *args, **kwargs)


class ZCAT19(ZCAT):
    def __init__(self, *args, **kwargs):
        super().__init__(19, *args, **kwargs)


class ZCAT20(ZCAT):
    def __init__(self, *args, **kwargs):
        super().__init__(20, *args, **kwargs)
