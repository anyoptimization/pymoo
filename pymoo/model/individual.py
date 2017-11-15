import numpy as np


class Individual:
    
    def __init__(self, x=None, f=None):
        self.x = x                   # design variables
        self.f = f                   # objective values
        self.g = None                   # constraint violations as vectors
        self.c = None                   # violation of constraints
        self.info = {}

    def evaluate(self, problem):
        self.f, self.g = problem.evaluate(self.x)
