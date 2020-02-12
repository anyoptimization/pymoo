class Result:
    """
    The resulting object of an optimization run.
    """

    def __init__(self) -> None:
        super().__init__()

        self.opt = None
        self.success = None
        self.message = None

        # ! other attributes to be set as well

        # the problem that was solved
        self.problem = None

        # the optimal solution for that problem
        self.pf = None

        # the algorithm that was used for optimization
        self.algorithm = None

        # the final population if it applies
        self.pop = None

        # directly the values of opt
        self.X, self.F, self.CV, self.G = None, None, None, None

        # all the timings that are stored of the run
        self.start_time, self.end_time, self.exec_time = None, None, None

        # the history of the optimization run is they were saved
        self.history = []
