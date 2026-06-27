"""Solution and solution set classes for backward compatibility."""

from pymoo.core.individual import Individual
from pymoo.core.population import Population


class Solution(Individual):
    """Backward-compatible alias for Individual."""


class SolutionSet(Population):
    """Backward-compatible alias for Population."""
