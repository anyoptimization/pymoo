import numpy as np

from pymoo.core.individual import calc_cv


class KKTPM:

    def calc(self, X, problem, ideal=None, utopian_eps=1e-4, rho=1e-3):
        """

        Returns the Karush-Kuhn-Tucker Approximate Measure.

        Parameters
        ----------
        X : np.array

        problem : pymoo.core.problem
        ideal : np.array
            The ideal point if not in the problem defined or intentionally overwritten.
        utopian_eps : float
            The epsilon used for decrease the ideal point to get the utopian point.
        rho : float
            Since augmented achievement scalarization function is used the F for all other weights
            - here rho - needs to be defined.

        Returns
        -------

        """

        # the final result to be returned
        kktpm = np.full((X.shape[0], 1), np.inf)
        fval = np.full((X.shape[0], 1), np.inf)

        # set the ideal point for normalization
        z = ideal

        # if not provided take the one defined in the problem
        if z is None:
            z = problem.ideal_point()
        z -= utopian_eps

        # for convenience get the counts directly
        n_solutions, n_var, n_obj, n_ieq_constr = X.shape[0], problem.n_var, problem.n_obj, problem.n_ieq_constr

        F, G, dF, dG = problem.evaluate(X, return_values_of=["F", "G", "dF", "dG"])
        CV = calc_cv(G=G)

        # loop through each solution to be considered
        for i in range(n_solutions):

            # get the corresponding values for this solution
            x, f, cv, df = X[i, :], F[i, :], CV[i], dF[i, :].swapaxes(1, 0)
            if n_ieq_constr > 0:
                g, dg = G[i, :], dG[i].T

            # if the solution that is provided is infeasible
            if cv > 0:
                _kktpm = 1 + cv
                _fval = None

            else:

                w = np.sqrt(np.sum(np.power(f - z, 2))) / (f - z)
                a_m = (df * w + (rho * np.sum(df * w, axis=1))[:, None]).T

                A = np.ones((problem.n_obj, problem.n_obj)) + a_m @ a_m.T
                b = np.ones(problem.n_obj)

                if n_ieq_constr > 0:
                    # a_j is just the transpose of the differential of constraints
                    a_j = dg.T

                    # part of the matrix for additional constraints
                    gsq = np.zeros((n_ieq_constr, n_ieq_constr))
                    np.fill_diagonal(gsq, g * g)

                    # now add the constraints to the optimization problem
                    A = np.vstack([np.hstack([A, a_m @ a_j.T]), np.hstack([a_j @ a_m.T, a_j @ a_j.T + gsq])])
                    b = np.hstack([b, np.zeros(n_ieq_constr)])

                method = "qr"
                u = solve(A, b, method=method)

                # until all the lagrange multiplier are positive
                while np.any(u < 0):

                    # go through one by one
                    for j in range(len(u)):

                        # if a lagrange multiplier is negative - we need to fix it
                        if u[j] < 0:
                            # modify the optimization problem
                            A[j, :], A[:, j], A[j, j] = 0, 0, 1
                            b[j] = 0

                            # resolve the problem and redefine u. for sure all preview u[j] are positive now
                            u = solve(A, b, method=method)

                # split up the lagrange multiplier for objective and not
                u_m, u_j = u[:n_obj], u[n_obj:]

                if n_ieq_constr > 0:
                    _kktpm = (1 - np.sum(u_m)) ** 2 + np.sum((np.vstack([a_m, a_j]).T @ u) ** 2)
                    _fval = _kktpm + np.sum((u_j * g.T) ** 2)
                else:
                    _kktpm = (1 - np.sum(u_m)) ** 2 + np.sum((a_m.T @ u) ** 2)
                    _fval = _kktpm

                ujgj = -g @ u_j
                if np.sum(u_m) + ujgj * (1 + ujgj) > 1:
                    adjusted_kktpm = - (u_j @ g.T)
                    projected_kktpm = (_kktpm * g @ g.T - g @ u_j) / (1 + g @ g.T)
                    _kktpm = (_kktpm + adjusted_kktpm + projected_kktpm) / 3

            # assign to the values to be returned
            kktpm[i] = _kktpm
            fval[i] = _fval

        return kktpm[:, 0]


def solve(A, b, method="elim"):
    if method == "elim":
        return np.linalg.solve(A, b)

    elif method == "qr":
        Q, R = np.linalg.qr(A)
        y = np.dot(Q.T, b)
        return np.linalg.solve(R, y)

    elif method == "svd":
        U, s, V = np.linalg.svd(A)  # SVD decomposition of A
        A_inv = np.dot(np.dot(V.T, np.linalg.inv(np.diag(s))), U.T)
        return A_inv @ b


if __name__ == '__main__':
    from pymoo.problems import get_problem
    from pymoo.gradient.automatic import AutomaticDifferentiation

    from pymoo.constraints.from_bounds import ConstraintsFromBounds
    problem = ConstraintsFromBounds(AutomaticDifferentiation(get_problem("zdt2", n_var=30)))

    # X = (0.5 * np.ones(10))[None, :]
    X = np.array(
        [0.394876, 0.963263, 0.173956, 0.126330, 0.135079, 0.505662, 0.021525, 0.947970, 0.827115, 0.015019, 0.176196,
         0.332064, 0.130997, 0.809491, 0.344737, 0.940107, 0.582014, 0.878832, 0.844734, 0.905392, 0.459880, 0.546347,
         0.798604, 0.285719, 0.490254, 0.599110, 0.015533, 0.593481, 0.433676, 0.807361])

    print(KKTPM().calc(X[None, :], problem))
