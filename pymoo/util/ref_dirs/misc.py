import numpy as np


def project_onto(y, p, n):
    """

    Project a point onto a plane.

    Parameters
    ----------
    y : The point that should be project
    p : A point which lies on the plane
    n : The normal vector of the plane

    Returns
    -------

    proj_y : The projection onto the plane

    """

    is_1d = False
    if len(y.shape) == 1:
        is_1d = True
        y = np.atleast_2d(y)

    # make sure the plane vector is normalized
    n = n / np.linalg.norm(n)

    proj_y = y - n[None, :] * ((y - p) @ n)[:, None]

    if is_1d:
        proj_y = proj_y[0]

    return proj_y


def project_onto_sum_equals_one_plane(y):
    n_dim = y.shape[-1]
    return project_onto(y, np.eye(n_dim)[0], np.ones(n_dim))


def project_onto_sum_equals_zero_plane(y):
    n_dim = y.shape[-1]
    return project_onto(y, np.zeros(n_dim), np.ones(n_dim))


def project_onto_unit_simplex(X):
    n_points, n_dim = X.shape

    # a boolean mask of violating values - less than 0
    I = X < 0

    # get all the points which are out of bounds and need to be fixed
    out_of_unit_simplex = np.where(I.sum(axis=1) > 0)[0]

    # now check for each point if it is still in bound
    for j in out_of_unit_simplex:

        # indices where the last point was already out of bounds
        subspace = np.logical_not(I[j])

        # project the bounds back onto the simplex in the subspace
        proj = np.zeros(n_dim)
        proj[subspace] = project_onto_sum_equals_one_plane(X[j][subspace])

        # set the result to the corresponding value
        X[j] = proj

        test1 = matrix_project_onto_sum_equals_one_plane(X[j][subspace])
        test2 = project_onto_sum_equals_one_plane(X[j][subspace])

        if not np.allclose(test1, test2):
            print("test")

        if np.any(X[j] < 0):
            print("test")


def project_onto_unit_simplex_recursive(X):
    # get all the points which are out of bounds and need to be fixed
    out_of_unit_simplex = np.where(np.any(X < 0, axis=1))[0]

    # now check for each point if it is still in bound
    for j in out_of_unit_simplex:

        while True:
            X[j, X[j] < 0] = 0

            # indices where the last point was already out of bounds
            subspace = X[j] > 0

            # project the bounds back onto the simplex in the subspace
            X[j, subspace] = project_onto_sum_equals_one_plane(X[j][subspace])

            if np.all(X[j] >= 0):
                break

    return X


def matrix_project_onto_sum_equals_one_plane(next):
    n_dim = len(next)
    P, S = np.eye(n_dim)[0], next

    # create for each subspace dimension a point on the hyperplane
    points = P + np.eye(n_dim)[1:]
    points /= points.sum(axis=1)[:, None]

    v = points - P
    s = S - P

    # solve a system of linear equations to project the point
    A = np.zeros((n_dim - 1, n_dim - 1))
    for i in range(n_dim - 1):
        for j in range(n_dim - 1):
            A[i, j] = np.dot(v[i], v[j])

    b = np.zeros(n_dim - 1)
    for i in range(n_dim - 1):
        b[i] = np.dot(s, v[i])

    x = np.linalg.solve(A, b)

    # finally calculate the projection onto the plane
    proj = (P + x @ v)

    return proj
