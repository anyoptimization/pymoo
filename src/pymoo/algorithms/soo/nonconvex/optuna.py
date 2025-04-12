from pymoo.util.optimum import filter_optimum

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.logging import get_logger
except:
    raise Exception("Please install optuna: pip install optuna")

from pymoo.core.algorithm import Algorithm
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.util.display.single import SingleObjectiveOutput


class Optuna(Algorithm):

    def __init__(self, sampler=None, output=SingleObjectiveOutput(), **kwargs):
        super().__init__(output=output, **kwargs)
        self.sampler = sampler

    def _setup(self, problem, **kwargs):

        sampler = self.sampler
        if sampler is None:
            sampler = TPESampler(seed=self.seed)

        # that disables the warning in the very beginning
        get_logger('optuna.storages._in_memory').disabled = True

        # create a new study
        self.study = optuna.create_study(study_name=f"Study@{id(self)}", sampler=sampler, direction='minimize')

        # the current trial for an individual
        self.trial = None

    def _infill(self):
        self.trial = self.study.ask()

        vars = self.problem.vars
        assert vars is not None, "Optuna needs explicitly defined variables."

        x = {}
        for name, param in vars.items():
            if isinstance(param, Real):
                lower, upper = param.bounds
                v = self.trial.suggest_float(name, lower, upper)
            elif isinstance(param, Integer):
                lower, upper = param.bounds
                v = self.trial.suggest_int(name, lower, upper)
            elif isinstance(param, Choice):
                options = param.options
                v = self.trial.suggest_categorical(name, options)
            elif isinstance(param, Binary):
                v = self.trial.suggest_categorical(name, [False, True])
            else:
                raise Exception("Type not supported yet.")
            x[name] = v

        return Individual(X=x)

    def _advance(self, infills=None, **kwargs):
        self.pop = Population.create(infills)
        self.study.tell(self.trial, infills.f)

    def _initialize_infill(self):
        return self._infill()

    def _initialize_advance(self, **kwargs):
        return self._advance(**kwargs)

    def _set_optimum(self):
        pop = self.pop
        if self.opt is not None:
            pop = Population.merge(self.opt, pop)
        self.opt = filter_optimum(pop, least_infeasible=True)



