from pymoo.model.termination import Termination


class ToleranceBasedTermination(Termination):

    def __init__(self,
                 tol=0.0025,
                 n_last=30,
                 nth_gen=5,
                 n_hist=None,
                 n_hist_at_least=1,
                 hist_of_metrics=False,
                 ) -> None:
        """

        Parameters
        ----------
        tol : float
            A tolerance value defined by the user that should be used for the concrete implementation

        n_last : int
            The last generations that should be considering during the calculations

        nth_gen : int
            Each n-th generation the termination should be checked for


        n_hist : int
            How much of the history should be kept in memory based on a sliding window.

        n_hist_at_least : int
            How much information of the history is necessary to calculate the metric.

        """
        super().__init__()

        # a tolerance value defined by the user
        self.tol = tol

        # each nth generation considering the n_last iterations
        self.nth_gen = nth_gen
        self.n_last = n_last

        # store the history to have a window of iterations
        self.history = []

        # the implementation can define how many information of the history is required for the calculation
        self.n_hist = max(n_hist, n_hist_at_least) if n_hist is not None else None
        self.n_hist_at_least = n_hist_at_least

        # the metric used for deciding to terminate or not - can be utilized by the concrete implementation
        self.metrics = []

        self.hist_of_metrics = hist_of_metrics
        self.hist_metrics = []

    def _do_continue(self, algorithm):

        # let the implementation extract the necessary information to be stored
        obj = self._store(algorithm)

        # store the current data in the history - truncate everything after window if enabled
        self.history.append(obj)
        if self.n_hist is not None:
            self.history = self.history[-self.n_hist:]

        # if there is enough history to calculate the metric
        if len(self.history) >= self.n_hist_at_least:

            # the implementation can calculate the metric based on history - it is truncated as well
            metric = self._calc_metric()

            # if desired keep a history of all metrics from the past - not only sliding window
            if self.hist_of_metrics:
                self.hist_metrics.append(metric)

            # now add to the window of metric and truncate if necessary
            self.metrics.append(metric)
            if self.n_last is not None:
                self.metrics = self.metrics[-self.n_last:]

        # if not enough metric data are collected (at least n_last) always continue
        is_nth_generation = algorithm.n_gen % self.nth_gen == 0

        # if both is given then let the implementation decide
        if is_nth_generation and len(self.metrics) >= self.n_hist_at_least:

            # ask the implementation whether to terminate or not
            return self._decide()

        # otherwise by default just continue
        else:
            return True

    # given an algorithm object decide what should be stored as historical information
    def _store(self, algorithm):
        return algorithm.opt

    # calculate a metric - this will be called each time - after the history is truncated
    def _calc_metric(self):
        return self.history[-1]

    # decide whether to continue or not
    def _decide(self):
        return True

    # returns the metric calculate in the current iteration - None if not calculated yet
    def metric(self):
        if self.metrics is not None and len(self.metrics) > 0:
            return self.metrics[-1]
        else:
            return None
