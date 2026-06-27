"""Univariate optimization algorithms."""

from pymoo.algorithms.soo.univariate.golden import GoldenSectionSearch
from pymoo.algorithms.soo.univariate.quadr_interp import QuadraticInterpolationSearch
from pymoo.algorithms.soo.univariate.wolfe import WolfeSearch

__all__ = [
    "GoldenSectionSearch",
    "QuadraticInterpolationSearch",
    "WolfeSearch",
]
