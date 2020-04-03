from pymoo.util.misc import to_numpy
from pymoo.util.termination.sliding_window_termination import SlidingWindowTermination


class SingleObjectiveSpaceToleranceTermination(SlidingWindowTermination):

    def __init__(self,
                 tol=1e-6,
                 n_last=20,
                 nth_gen=1,
                 n_max_gen=None,
                 n_max_evals=None,
                 **kwargs) -> None:
        super().__init__(metric_window_size=n_last,
                         data_window_size=2,
                         min_data_for_metric=2,
                         nth_gen=nth_gen,
                         n_max_gen=n_max_gen,
                         n_max_evals=n_max_evals,
                         **kwargs)
        self.tol = tol

    def _store(self, algorithm):
        return algorithm.opt.get("F").min()

    def _metric(self, data):
        last, current = data[-2], data[-1]
        return last - current

    def _decide(self, metrics):
        delta_f = to_numpy(metrics)
        return delta_f.max() > self.tol
