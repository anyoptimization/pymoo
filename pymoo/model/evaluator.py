class Evaluator:
    """

    The evaluator class which is used during the algorithm execution to limit the number of evaluations.
    This can be based on convergence, maximum number of evaluations, or other criteria.

    """

    def __init__(self, n_eval=1e10):
        self.n_max_eval = n_eval
        self.n_eval = 0

    def eval(self, problem, X, **kwargs):
        """

        This function is used to return the result of one valid evaluation.

        Parameters
        ----------
        problem : class
            The problem which is used to be evaluated
        X : np.ndarray
            A two dimensional array where each row is a variable to evaluate
        kwargs : dict
            Additional arguments which might be necessary for the problem to evaluate.

        Returns
        -------
        val : any
            Returns whatever the problem used to evaluate returns. Can be only objective values or also constraints.

        """
        if len(X.shape) == 1:
            self.n_eval += 1
        else:
            self.n_eval += X.shape[0]
        return problem.evaluate(X, **kwargs)

    def n_remaining(self):
        """

        Returns
        -------
        n_remaining : int
            The number of function evaluations that are left.

        """
        return self.n_max_eval - self.n_eval

    def has_remaining(self):
        """

        Returns
        -------
        has_remaining : bool
            True if function evaluations are left and false otherwise

        """
        return self.n_eval < self.n_max_eval
