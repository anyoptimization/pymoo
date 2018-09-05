from pymoo.indicators.igd import IGD


class Termination:

    def do_continue(self, D):
        return not self.has_finished(D)

    def has_finished(self, D):
        return not self.do_continue(D)


class MaximumFunctionCallTermination(Termination):

    def __init__(self, n_max_evals) -> None:
        super().__init__()
        self.n_max_evals = n_max_evals

    def do_continue(self, D):
        return D['evaluator'].n_eval < self.n_max_evals


class MaximumGenerationTermination(Termination):

    def __init__(self, n_max_gen) -> None:
        super().__init__()
        self.n_max_gen = n_max_gen

    def do_continue(self, D):
        return D['n_gen'] < self.n_max_gen


class IGDTermination(Termination):

    def __init__(self, problem, igd) -> None:
        super().__init__()
        pf = problem.pareto_front()

        if pf is None:
            raise Exception("You can only use IGD termination criteria if the pareto front is known!")

        self.obj = IGD(pf)
        self.igd = igd

    def do_continue(self, D):
        return self.obj.calc(D['pop'].F) > self.igd
