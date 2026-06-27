"""Module containing infrastructure for representing decision variable classes."""

from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from pymoo.util import default_random_state

__all__ = [
    "Variable",
    "BoundedVariable",
    "Real",
    "Integer",
    "Binary",
    "Choice",
    "get",
]


class Variable:
    """Semi-abstract base class for the representation of a decision variable."""

    def __init__(
        self,
        value: Optional[object] = None,
        active: bool = True,
        flag: str = "default",
    ) -> None:
        """Initialize Variable.

        Args:
            value: Value the decision variable is to take.
            active: Whether the variable is active or inactive.
            flag: Flag to bind to the decision variable.
        """
        super().__init__()
        self.value = value
        self.flag = flag
        self.active = active

    @default_random_state
    def sample(
        self,
        n: Optional[int] = None,
        random_state: object = None,
    ) -> Union[object, np.ndarray]:
        """Randomly sample n instances of a decision variable.

        Args:
            n: Number of samples. If None, sample a single variable.
            random_state: Random state for reproducibility.

        Returns:
            If n is int, array of shape (n,). If n is None, single object.
        """
        if n is None:
            return self._sample(1, random_state=random_state)[0]
        else:
            return self._sample(n, random_state=random_state)

    def _sample(
        self,
        n: int,
        random_state: object = None,
    ) -> np.ndarray:
        """Randomly sample n instances (abstract method)."""
        pass

    def set(
        self,
        value: object,
    ) -> None:
        """Set the value of a decision variable.

        Args:
            value: Value to assign to the decision variable.
        """
        self.value = value

    def get(self, **kwargs: Any) -> object:
        """Get the value of a decision variable.

        Args:
            kwargs: Additional keyword arguments.

        Returns:
            The value of the decision variable.
        """
        return self.value


class BoundedVariable(Variable):
    """Semi-abstract class for the representation of a bounded decision variable."""

    def __init__(
        self,
        value: Optional[object] = None,
        bounds: Tuple[Optional[int | float], Optional[int | float]] = (None, None),
        strict: Optional[Tuple[Optional[object], Optional[object]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize BoundedVariable.

        Args:
            value: Value the decision variable is to take.
            bounds: Tuple of (lower, upper) limits for the variable.
            strict: Strict boundaries for the variable.
            kwargs: Additional keyword arguments for active and flag.
        """
        super().__init__(value=value, **kwargs)
        self.bounds = bounds

        if strict is None:
            strict = bounds
        self.strict = strict

    @property
    def lb(self) -> object:
        """Lower bound of the decision variable.

        Returns:
            The decision variable lower bound.
        """
        return self.bounds[0]

    @property
    def ub(self) -> object:
        """Upper bound of the decision variable.

        Returns:
            The decision variable upper bound.
        """
        return self.bounds[1]


class Real(BoundedVariable):
    """Class for the representation of bounded, real decision variables."""

    vtype = float

    def _sample(
        self,
        n: int,
        random_state: object = None,
    ) -> np.ndarray:
        """Randomly sample n instances of a real, bounded decision variable.

        Args:
            n: Number of decision variable samples to draw.
            random_state: Random state for sampling.

        Returns:
            Array of shape (n,) containing sampled real variables.
        """
        low, high = self.bounds
        return random_state.uniform(low=low, high=high, size=n)


class Integer(BoundedVariable):
    """Class for the representation of bounded, integer decision variables."""

    vtype = int

    def _sample(
        self,
        n: int,
        random_state: object = None,
    ) -> np.ndarray:
        """Randomly sample n instances of a bounded, integer decision variable.

        Args:
            n: Number of decision variable samples to draw.
            random_state: Random state for sampling.

        Returns:
            Array of shape (n,) containing sampled integer variables.
        """
        low, high = self.bounds
        assert high is not None, "Integer variable requires an upper bound"
        return random_state.integers(low, high + 1, size=n)


class Binary(BoundedVariable):
    """Class for the representation of a binary, bounded decision variable."""

    vtype = bool

    def _sample(
        self,
        n: int,
        random_state: object = None,
    ) -> np.ndarray:
        """Randomly sample n instances of a bounded, binary decision variable.

        Args:
            n: Number of decision variable samples to draw.
            random_state: Random state for sampling.

        Returns:
            Array of shape (n,) containing sampled binary variables.
        """
        return random_state.random(size=n) < 0.5


class Choice(Variable):
    """Class for the representation of a discrete, subset decision variable."""

    vtype = object

    def __init__(
        self,
        value: Optional[object] = None,
        options: Optional[ArrayLike] = None,
        all: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Choice.

        Args:
            value: Value the decision variable is to take.
            options: List of decision variable options to choose from.
            all: Strict list of options. If None, copied from options.
            kwargs: Additional keyword arguments for active and flag.
        """
        super().__init__(value=value, **kwargs)
        self.options = options

        if all is None:
            all = options
        self.all = all

    def _sample(
        self,
        n: int,
        random_state: object = None,
    ) -> np.ndarray:
        """Randomly sample n instances of a discrete decision variable.

        Args:
            n: Number of decision variable samples to draw.
            random_state: Random state for sampling.

        Returns:
            Array of shape (n,) containing sampled choices.
        """
        return random_state.choice(self.options, size=n)


def get(
    *args: Tuple[Union[Variable, object], ...],
    size: Optional[Union[tuple, int]] = None,
    **kwargs: Any,
) -> Union[tuple, object, None]:
    """Get decision variable values from a tuple of Variable objects.

    Args:
        args: Tuple of Variable or objects.
        size: Size to reshape decision variables.
        kwargs: Additional keyword arguments for get method.

    Returns:
        Decision variable value(s).
    """
    if len(args) == 0:
        return None

    ret = []
    for arg in args:
        v = arg.get(**kwargs) if isinstance(arg, Variable) else arg

        if size is not None:
            if isinstance(v, np.ndarray):
                v = np.reshape(v, size)
            else:
                v = np.full(size, v)

        ret.append(v)

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
