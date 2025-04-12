# The code is translated from Matlab by ahcheriet@gmail.com
# ========================================================|#
# The 14 test functions are for cec2018 competition on    |
# dynamic multiobjective optimisation. This document is   |
# free to disseminate for academic use.                   |
# --------------------------------------------------------|#
# The "time" term in the test suite is defined as:        |
#          t=1/nt*floor(tau/taut)                         |
# where - nt:    severity of change                       |
#       - taut:  frequency of change                      |
#       - tau:   current generation counter               |
# --------------------------------------------------------|#
# Any questions can be directed to                        |
#    Dr. Shouyong Jiang at math4neu@gmail.com.            |
#                                                         |
# ========================================================|#

# cec2018_DF(probID, x, tau, taut, nt)
# INPUT:
#       probID: test problem identifier (i.e. 'DF1')
#       x:      variable vector
#       tau:    current generation counter
#       taut:   frequency of change
#       nt:     severity of change
#
# OUTPUT:
#       f:      objective vector
#
from numpy import power, setdiff1d, exp, prod


def get_bounds(problem_id='DF1', n_vars=10):
    lx = np.zeros(n_vars)
    ux = np.ones(n_vars)
    if problem_id == 'DF3':
        lx = -1 * np.ones(n_vars)
        ux = 2 * np.ones(n_vars)
        lx[0] = 0.0
        ux[0] = 1.0
    if problem_id == 'DF4':
        lx = -2 * np.ones(n_vars)
        ux = -2 * np.ones(n_vars)
    if problem_id == 'DF5':
        lx = -1 * np.ones(n_vars)
        ux = -1 * np.ones(n_vars)
        lx[0] = 0.0
        ux[0] = 1.0
    if problem_id == 'DF6':
        lx = -1 * np.ones(n_vars)
        ux = -1 * np.ones(n_vars)
        lx[0] = 0.0
        ux[0] = 1.0
    if problem_id == 'DF7':
        lx = 0 * np.ones(n_vars)
        ux = 1 * np.ones(n_vars)
        lx[0] = 1.0
        ux[0] = 4.0
    if problem_id == 'DF8':
        lx = -1 * np.ones(n_vars)
        ux = -1 * np.ones(n_vars)
        lx[0] = 0.0
        ux[0] = 1.0
    if problem_id == 'DF9':
        lx = -1 * np.ones(n_vars)
        ux = -1 * np.ones(n_vars)
        lx[0] = 0.0
        ux[0] = 1.0
    if problem_id == 'DF10':
        lx = -1 * np.ones(n_vars)
        ux = -1 * np.ones(n_vars)
        lx[0] = 0.0
        lx[1] = 0.0
        ux[0] = 1.0
        ux[1] = 1.0
    if problem_id == 'DF12':
        lx = -1 * np.ones(n_vars)
        ux = -1 * np.ones(n_vars)
        lx[0] = 0.0
        lx[1] = 0.0
        ux[0] = 1.0
        ux[1] = 1.0
    if problem_id == 'DF13':
        lx = -1 * np.ones(n_vars)
        ux = -1 * np.ones(n_vars)
        lx[0] = 0.0
        lx[1] = 0.0
        ux[0] = 1.0
        ux[1] = 1.0
    if problem_id == 'DF14':
        lx = -1 * np.ones(n_vars)
        ux = -1 * np.ones(n_vars)
        lx[0] = 0.0
        lx[1] = 0.0
        ux[0] = 1.0
        ux[1] = 1.0
    return lx, ux


def cec2018_DF(problemID='DF1', x=None, t=None):
    # INPUT:
    #       probID: test problem identifier (i.e. 'DF1')
    #       x:      variable vector
    #       tau:    current generation counter
    #       taut:   frequency of change
    #       nt:     severity of change

    # OUTPUT:
    #       f:      objective vector

    # the first change occurs after T0 generations, that is, the
    # generation at which a change occurs is (T0+1), (T0+taut+1), etc.

    T0 = 50
    # calculate time instant
    n = len(x)
    f = {}
    if problemID == 'DF1':
        G = abs(sin(0.5 * pi * t))
        H = 0.75 * sin(0.5 * pi * t) + 1.25
        g = 1 + sum((x[1:] - G) ** 2)
        f[0] = x[0]
        f[1] = g * power(1 - (x[0] / g), H)
    if problemID == 'DF2':
        G = abs(sin(0.5 * pi * t))
        r = 1 + floor((n - 1) * G)
        tmp = setdiff1d(range(0, n), [int(r)])
        g = 1 + sum([(x[int(index)] - G) ** 2 for index in tmp])
        f[0] = x[int(r - 1)]
        f[1] = g * (power(1 - (x[int(r - 1)] / g), 0.5))
    if problemID == 'DF3':
        G = sin(0.5 * pi * t)
        H = G + 1.5
        g = 1 + sum(power(x[1:] - G - x[0], H) ** 2)
        f[0] = x[0]
        f[1] = g * power(1 - (x[0] / g), H)
    if problemID == 'DF4':
        a = sin(0.5 * pi * t)
        b = 1 + abs(cos(0.5 * pi * t))
        H = 1.5 + a
        g = 1 + sum((x[1:] - a * x[0] ** 2 / x[1:]) ** 2)
        f[0] = g * power(abs(x[0] - a), H)
        f[1] = g * power(abs(x[0] - a - b), H)
    if problemID == 'DF5':
        G = sin(0.5 * pi * t)
        w = floor(10 * G)
        g = 1 + sum((x[1:] - G) ** 2)
        f[0] = g * (x[0] + 0.02 * sin(w * pi * x[0]))
        f[1] = g * (1 - x[0] + 0.02 * sin(w * pi * x[0]))
    if problemID == 'DF6':
        G = sin(0.5 * pi * t)
        a = 0.2 + 2.8 * abs(G)
        y = x[1:] - G
        g = 1 + sum((abs(G) * y ** 2 - 10 * cos(2 * pi * y) + 10))
        f[0] = g * power(x[0] + 0.1 * sin(3 * pi * x[0]), a)
        f[1] = g * power(1 - x[0] + 0.1 * sin(3 * pi * x[0]), a)
    if problemID == 'DF7':
        a = 5 * cos(0.5 * pi * t)
        tmp = 1 / (1 + exp(a * (x[0] - 2.5)))
        g = 1 + sum(power(x[1:] - tmp, 2))
        f[0] = g * (1 + t) / x[0]
        f[1] = g * x[0] / (1 + t)
    if problemID == 'DF8':
        G = sin(0.5 * pi * t)
        a = 2.25 + 2 * cos(2 * pi * t)
        b = 100 * G ** 2
        tmp = G * sin(power(4 * pi * x[0], b)) / (1 + abs(G))
        g = 1 + sum((x[1:] - tmp) ** 2)
        f[0] = g * (x[0] + 0.1 * sin(3 * pi * x[0]))
        f[1] = g * power(1 - x[1] + 0.1 * sin(3 * pi * x[1]), a)
    if problemID == 'DF9':
        N = 1 + floor(10 * abs(sin(0.5 * pi * t)))
        g = 1
        for i in range(1, n):
            tmp = x[i] - cos(4 * t + x[0] + x[i - 1])
            g = g + tmp ** 2
        f[0] = g * (x[0] + max(0, (0.1 + 0.5 / N) * sin(2 * N * pi * x[0])))
        f[1] = g * (1 - x[0] + max(0, (0.1 + 0.5 / N) * sin(2 * N * pi * x[0])))
    if problemID == 'DF10':
        G = sin(0.5 * pi * t)
        H = 2.25 + 2 * cos(0.5 * pi * t)
        tmp = sin(2 * pi * (x[0] + x[1])) / (1 + abs(G))
        g = 1 + sum((x[2:] - tmp) ** 2)
        f[0] = g * power(sin(0.5 * pi * x[0]), H)
        f[1] = g * power(sin(0.5 * pi * x[1]), H) * power(cos(0.5 * pi * x[0]), H)
        f[2] = g * power(cos(0.5 * pi * x[1]), H) * power(cos(0.5 * pi * x[0]), H)
    if problemID == 'DF11':
        G = abs(sin(0.5 * pi * t))
        g = 1 + G + sum((x[2:] - 0.5 * G * x[0]) ** 2)
        y = [pi * G / 6.0 + (pi / 2 - pi * G / 3.0) * x[i] for i in [0, 1]]
        f[0] = g * sin(y[0])
        f[1] = g * sin(y[1]) * cos(y[0])
        f[2] = g * cos(y[1]) * cos(y[0])
    if problemID == 'DF12':
        k = 10 * sin(pi * t)
        tmp1 = x[2:] - sin(t * x[0])
        tmp2 = [sin(floor(k * (2 * x[0] - 1)) * pi / 2)]
        g = 1 + sum(tmp1 ** 2) + prod(tmp2)
        f[0] = g * cos(0.5 * pi * x[1]) * cos(0.5 * pi * x[0])
        f[1] = g * sin(0.5 * pi * x[1]) * cos(0.5 * pi * x[0])
        f[2] = g * sin(0.5 * pi * x[1])
    if problemID == 'DF13':
        G = sin(0.5 * pi * t);
        p = floor(6 * G);
        g = 1 + sum((x[2:] - G) ** 2)
        f[0] = g * cos(0.5 * pi * x[0]) ** 2
        f[1] = g * cos(0.5 * pi * x[1]) ** 2
        f[2] = g * sin(0.5 * pi * x[0]) ** 2 + sin(0.5 * pi * x[0]) * cos(p * pi * x[0]) ** 2 + sin(
            0.5 * pi * x[1]) ** 2 + sin(0.5 * pi * x[1]) * cos(p * pi * x[1]) ** 2
    if problemID == 'DF14':
        G = sin(0.5 * pi * t)
        g = 1 + sum((x[2:] - G) ** 2)
        y = 0.5 + G * (x[0] - 0.5)
        f[0] = g * (1 - y + 0.05 * sin(6 * pi * y))
        f[1] = g * (1 - x[1] + 0.05 * sin(6 * pi * x[1])) * (y + 0.05 * sin(6 * pi * y))
        f[2] = g * (x[1] + 0.05 * sin(6 * pi * x[1])) * (y + 0.05 * sin(6 * pi * y))
    return f


import numpy as np
from numpy import pi, dot, floor, sin, cos, multiply, arange
from copy import copy

false = False
true = True


# cec2018_pf.m

# ========================================================|#
# PF calculation for 14 cec2018 test functions on         |
# dynamic multiobjective optimisation. This document is   |
# free to disseminate for academic use.                   |
# --------------------------------------------------------|#
# The "time" term in the test suite is defined as:        |
#          t=1/nt*floor(tau/taut)                         |
# where - nt:    severity of change                       |
#       - taut:  frequency of change                      |
#       - tau:   current generation counter               |
# --------------------------------------------------------|#
# Any questions can be directed to                        |
#    Dr. Shouyong Jiang at math4neu@gmail.com.            |
#                                                         |
# ========================================================|#


def cec2018_DF_PF(probID=None, t=1, n_points=100, *args, **kwargs):
    # INPUT:
    #       probID: test problem identifier (i.e. 'DF1')
    #       tau:    current generation counter
    #       taut:   frequency of change
    #       nt:     severity of change

    # OUTPUT:
    #       h:      nondominated solutions

    T0 = 50
    g = 1
    H = 50

    if 'DF1' == (probID):
        x = np.linspace(0, 1, n_points)
        H = dot(0.75, sin(dot(dot(0.5, pi), t))) + 1.25
        f1 = copy(x)
        f2 = dot(g, (1 - (x / g) ** H))
        h = get_PF(np.array([f1, f2]), false)
    if 'DF2' == (probID):
        x = np.linspace(0, 1, n_points)
        G = abs(sin(dot(dot(0.5, pi), t)))
        f1 = copy(x)
        f2 = dot(g, (1 - (x / g) ** 0.5))  # To be sure
        h = get_PF(np.array([f1, f2]), false)
    if 'DF3' == (probID):
        x = np.linspace(0, 1, n_points)
        G = sin(dot(dot(0.5, pi), t))
        H = G + 1.5
        f1 = copy(x)
        f2 = dot(g, (1 - (x / g) ** H))
        h = get_PF(np.array([f1, f2]), false)
    if 'DF4' == (probID):
        a = sin(dot(dot(0.5, pi), t))
        b = 1 + abs(cos(dot(dot(0.5, pi), t)))
        x = np.linspace(a, a + b)
        H = 1.5 + a
        f1 = dot(g, abs(x - a) ** H)
        f2 = dot(g, abs(x - a - b) ** H)  # Maybe
        h = get_PF(np.array([f1, f2]), false)
    if 'DF5' == (probID):
        x = np.linspace(0, 1, n_points)
        G = sin(dot(dot(0.5, pi), t))
        w = floor(dot(10, G))
        f1 = dot(g, (x + dot(0.02, sin(dot(dot(w, pi), x)))))
        f2 = dot(g, (1 - x + dot(0.02, sin(dot(dot(w, pi), x)))))
        h = get_PF(np.array([f1, f2]), false)
    if 'DF6' == (probID):
        x = np.linspace(0, 1, n_points)
        G = sin(dot(dot(0.5, pi), t))
        a = 0.2 + dot(2.8, abs(G))
        f1 = dot(g, (x + dot(0.1, sin(dot(dot(3, pi), x)))) ** a)
        f2 = dot(g, (1 - x + dot(0.1, sin(dot(dot(3, pi), x)))) ** a)
        h = get_PF(np.array([f1, f2]), false)
    if 'DF7' == (probID):
        x = np.linspace(1, 4, n_points)
        f1 = dot(g, (1 + t)) / x
        f2 = dot(g, x) / (1 + t)
        h = get_PF(np.array([f1, f2]), false)
    if 'DF8' == (probID):
        x = np.linspace(0, 1, n_points)
        a = 2.25 + dot(2, cos(dot(dot(2, pi), t)))
        f1 = dot(g, (x + dot(0.1, sin(dot(dot(3, pi), x)))))
        f2 = dot(g, (1 - x + dot(0.1, sin(dot(dot(3, pi), x)))) ** a)
        h = get_PF(np.array([f1, f2]), false)
    if 'DF9' == (probID):
        x = np.linspace(0, 1, n_points)
        #        N = 1 + floor(dot(10, abs(sin(dot(dot(0.5, pi), t)))))

        N = 1 + floor(10 * abs(sin(0.5 * pi * t)))
        print(max(0, max((0.1 + 0.5 / N) * sin(2 * N * pi * x))))
        f1 = g * (x + max(0, max((0.1 + 0.5 / N) * sin(2 * N * pi * x))))
        f2 = g * (1 - x + max(0, max((0.1 + 0.5 / N) * sin(2 * N * pi * x))))

        #        f1 = dot(g, (x + max(0, dot((0.1 + 0.5 / N), sin(dot(dot(dot(2, N), pi), x))))))
        #        f2 = dot(g, (1 - x + max(0, dot((0.1 + 0.5 / N), sin(dot(dot(dot(2, N), pi), x))))))
        h = get_PF(np.array([f1, f2]), true)
    if 'DF10' == (probID):
        x1, x2 = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, H), indexing='xy')
        H = 2.25 + dot(2, cos(dot(dot(0.5, pi), t)))
        f1 = dot(g, sin(dot(dot(0.5, pi), x1)) ** H)
        f2 = multiply(dot(g, sin(dot(dot(0.5, pi), x2)) ** H),
                      cos(dot(dot(0.5, pi), x1)) ** H)
        f3 = multiply(dot(g, cos(dot(dot(0.5, pi), x2)) ** H),
                      cos(dot(dot(0.5, pi), x1)) ** H)
        h = get_PF(np.array([f1, f2, f3]), false)
    if 'DF11' == (probID):
        x1, x2 = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, H), indexing='xy')
        G = abs(sin(dot(dot(0.5, pi), t)))
        y1 = dot(pi, G) / 6 + dot((pi / 2 - dot(pi, G) / 3), x1)
        y2 = dot(pi, G) / 6 + dot((pi / 2 - dot(pi, G) / 3), x2)
        f1 = multiply(g, sin(y1))
        f2 = dot(multiply(g, sin(y2)), cos(y1))
        f3 = dot(multiply(g, cos(y2)), cos(y1))
        h = get_PF(np.array([f1, f2, f3]), false)
    if 'DF12' == (probID):
        x1, x2 = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, H), indexing='xy')
        k = dot(10, sin(dot(pi, t)))
        tmp2 = abs(
            multiply(sin(dot(floor(dot(k, (dot(2, x1) - 1))), pi) / 2),
                     sin(dot(floor(dot(k, (dot(2, x2) - 1))), pi) / 2)))
        g = 1 + tmp2
        f1 = multiply(multiply(g, cos(dot(dot(0.5, pi), x2))),
                      cos(dot(dot(0.5, pi), x1)))
        f2 = multiply(multiply(g, sin(dot(dot(0.5, pi), x2))),
                      cos(dot(dot(0.5, pi), x1)))
        f3 = multiply(g, sin(dot(dot(0.5, pi), x1)))
        h = get_PF(np.array([f1, f2, f3]), true)
    if 'DF13' == (probID):
        x1, x2 = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, H), indexing='xy')
        G = sin(dot(dot(0.5, pi), t))
        p = floor(dot(6, G))
        f1 = multiply(g, cos(dot(dot(0.5, pi), x1)) ** 2)
        f2 = multiply(g, cos(dot(dot(0.5, pi), x2)) ** 2)
        f3 = multiply(g, sin(dot(dot(0.5, pi), x1)) ** 2) + multiply(
            sin(dot(dot(0.5, pi), x1)),
            cos(dot(dot(p, pi), x1)) ** 2) + sin(
            dot(dot(0.5, pi), x2)) ** 2 + multiply(
            sin(dot(dot(0.5, pi), x2)), cos(dot(dot(p, pi), x2)) ** 2)
        h = get_PF(np.array([f1, f2, f3]), true)
    if 'DF14' == (probID):
        x1, x2 = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, H), indexing='xy')
        G = sin(dot(dot(0.5, pi), t))
        y = 0.5 + dot(G, (x1 - 0.5))
        f1 = multiply(g, (1 - y + dot(0.05, sin(dot(dot(6, pi), y)))))
        f2 = multiply(multiply(g, (
                1 - x2 + dot(0.05, sin(dot(dot(6, pi), x2))))),
                      (y + dot(0.05, sin(dot(dot(6, pi), y)))))
        f3 = multiply(
            multiply(g, (x2 + dot(0.05, sin(dot(dot(6, pi), x2))))),
            (y + dot(0.05, sin(dot(dot(6, pi), y)))))
        h = get_PF(np.array([f1, f2, f3]), false)
    return h


def get_PF(f=None, nondominate=None, *args, **kwargs):
    ncell = len(f)
    s = np.size(f[1])
    h = []
    for i in arange(ncell):
        fi = np.reshape(f[i], s, order='F')
        h.append(fi)
    h = np.array(h).T
    h = np.reshape(h, (s, ncell))

    if nondominate:
        print("Run Non dominating Sorting")
        h = []
        pass
    #     in_ = get_skyline(h)
    #     h = h(in_, arange())
    return h
