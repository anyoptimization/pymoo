import numpy as np

from pymoo.performance_indicator.gd import GD
from pymoo.performance_indicator.igd import IGD
from pymoo.performance_indicator.hv import Hypervolume


def pareto_front_if_possible(problem):
    try:
        return problem.pareto_front()
    except:
        return None


class Output:

    def __init__(self, default_width=12) -> None:
        super().__init__()
        self.default_width = default_width
        self.attrs = []

    def append(self, name, number, format_if_float=True, width=None):
        if width is None:
            width = self.default_width
        if format_if_float and isinstance(number, float):
            number = self.format_float(number, width)

        self.attrs.append([name, number, width])

    def format_float(self, number, width):
        if number >= 10 or number * 1e5 < 1:
            return f"%.{width - 7}E" % number
        else:
            return f"%.{width - 3}f" % number

    def extend(self, *args):
        for arg in args:
            self.append(*arg)

    def clear(self):
        self.attrs = []

    def create_regex(self):
        return " | ".join(["{}"] * len(self.attrs))

    def do(self):
        if len(self.attrs) > 0:
            regex = self.create_regex()
            val = regex.format(*[str(val).rjust(width) for _, val, width in self.attrs])
            print(val)

    def header(self):
        regex = self.create_regex()
        s = regex.format(*[str(name).center(width) for name, _, width in self.attrs])
        print("=" * len(s))
        print(s)
        print("=" * len(s))


class Display:

    def __init__(self, output=None, attributes=None):
        super().__init__()
        self.output = output
        if self.output is None:
            self.output = Output()

        self.display_header = True
        self.pareto_front_is_available = None
        self.pf = None
        self.attributes = attributes

    def do(self, problem, evaluator, algorithm, pf=None):

        try:

            # if pf is false nothing happens, otherwise we try to get the pareto front from the problem
            if pf is None or (isinstance(pf, bool) and pf):

                # if we have not tried it before
                if self.pareto_front_is_available is None:
                    # see if you can get it and set the boolean for the future
                    self.pf = pareto_front_if_possible(problem)
                    self.pareto_front_is_available = self.pf is not None

            self.output.clear()

            # get the actual attributes to display
            self._do(problem, evaluator, algorithm)

            if self.attributes is not None:
                self.output.attrs = [attr for attr in self.output.attrs if attr[0] in self.attributes]

            if self.display_header:
                self.output.header()

            # print the actually line
            self.output.do()

        # catch any exception to make sure the algorithm does not fail because of printing
        except:
            print("WARNING: Error while preparing the output to be printed.")

        self.display_header = False

    def _do(self, problem, evaluator, algorithm):
        self.output.extend(*[('n_gen', algorithm.n_gen, False, 5), ('n_eval', evaluator.n_eval, False, 7)])


class SingleObjectiveDisplay(Display):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)

        F, CV, feasible = algorithm.pop.get("F", "CV", "feasible")
        feasible = np.where(feasible[:, 0])[0]

        if problem.n_constr > 0:
            self.output.append("cv (min)", CV.min())
            self.output.append("cv (avg)", np.mean(CV))

        if len(feasible) > 0:
            _F = F[feasible]
            self.output.append("favg", np.mean(_F))
            self.output.append("fopt", np.min(_F))
        else:
            self.output.extend(*[('favg', "-"), ('fopt', "-")])


class MultiObjectiveDisplay(Display):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)

        F, CV, feasible = algorithm.pop.get("F", "CV", "feasible")
        feasible = np.where(feasible[:, 0])[0]

        if problem.n_constr > 0:
            self.output.append("cv (min)", CV.min())
            self.output.append("cv (avg)", np.mean(CV))

        if len(feasible) > 0:
            if self.pareto_front_is_available:
                _F = F[feasible]
                self.output.append("igd", IGD(self.pf).calc(_F))
                self.output.append("gd", GD(self.pf).calc(_F))
                if problem.n_obj == 2:
                    self.output.append("hv", Hypervolume(pf=self.pf).calc(_F))
        else:
            if self.pareto_front_is_available:
                self.output.extend(*[('igd', "-"), ('gd', "-")])
                if problem.n_obj == 2:
                    self.output.append("hv", "-")
