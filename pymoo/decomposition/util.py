import autograd.numpy as anp


def calc_distance_to_weights(F, weights, utopian_point):
    norm = anp.linalg.norm(weights, axis=1)
    F = F - utopian_point

    d1 = (F * weights).sum(axis=1) / norm
    d2 = anp.linalg.norm(F - (d1[:, None] * weights / norm[:, None]), axis=1)

    return d1, d2
