from abc import abstractmethod

from pymoo.util.sliding_window import SlidingWindow
from pymoo.util.termination.collection import TerminationCollection
from pymoo.util.termination.max_eval import MaximumFunctionCallTermination
from pymoo.util.termination.max_gen import MaximumGenerationTermination


class SlidingWindowTermination(TerminationCollection):

    def __init__(self,
                 metric_window_size=None,
                 data_window_size=None,
                 min_data_for_metric=1,
                 nth_gen=1,
                 n_max_gen=None,
                 n_max_evals=None,
                 truncate_metrics=True,
                 truncate_data=True,
                 ) -> None:
        """

        Parameters
        ----------

        metric_window_size : int
            The last generations that should be considering during the calculations

        data_window_size : int
            How much of the history should be kept in memory based on a sliding window.

        nth_gen : int
            Each n-th generation the termination should be checked for

        """

        super().__init__(MaximumGenerationTermination(n_max_gen=n_max_gen),
                         MaximumFunctionCallTermination(n_max_evals=n_max_evals))

        # the window sizes stored in objects
        self.data_window_size = data_window_size
        self.metric_window_size = metric_window_size

        # the obtained data at each iteration
        self.data = SlidingWindow(data_window_size) if truncate_data else []

        # the metrics calculated also in a sliding window
        self.metrics = SlidingWindow(metric_window_size) if truncate_metrics else []

        # each n-th generation the termination decides whether to terminate or not
        self.nth_gen = nth_gen

        # number of entries of data need to be stored to calculate the metric at all
        self.min_data_for_metric = min_data_for_metric

    def _do_continue(self, algorithm):

        # if the maximum generation or maximum evaluations say terminated -> do so
        if not super()._do_continue(algorithm):
            return False

        # store the data decided to be used by the implementation
        obj = self._store(algorithm)
        if obj is not None:
            self.data.append(obj)

        # if enough data has be stored to calculate the metric
        if len(self.data) >= self.min_data_for_metric:
            metric = self._metric(self.data[-self.data_window_size:])
            if metric is not None:
                self.metrics.append(metric)

        # if its the n-th generation and enough metrics have been calculated make the decision
        if algorithm.n_gen % self.nth_gen == 0 and len(self.metrics) >= self.metric_window_size:

            # ask the implementation whether to terminate or not
            return self._decide(self.metrics[-self.metric_window_size:])

        # otherwise by default just continue
        else:
            return True

    # given an algorithm object decide what should be stored as historical information - by default just opt
    def _store(self, algorithm):
        return algorithm.opt

    @abstractmethod
    def _decide(self, metrics):
        pass

    @abstractmethod
    def _metric(self, data):
        pass

    def get_metric(self):
        if len(self.metrics) > 0:
            return self.metrics[-1]
        else:
            return None
