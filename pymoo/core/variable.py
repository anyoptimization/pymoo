"""
Module containing infrastructure for representing decision variable classes.
"""

# public API for when using ``from pymoo.core.variable import *``
__all__ = [
    "Variable",
    "BoundedVariable",
    "Real",
    "Integer",
    "Binary",
    "Choice",
    "get",
]

from typing import Any, Optional, Tuple
from typing import Union
import numpy as np
from numpy.typing import ArrayLike
from pymoo.util import default_random_state


class Variable(object):
    """
    Semi-abstract base class for the representation of a decision variable.
    """

    def __init__(
            self, 
            value: Optional[object] = None, 
            active: bool = True, 
            flag: str = "default",
        ) -> None:
        """
        Constructor for the ``Variable`` class.

        Parameters
        ----------
        value : object, None
            Value the decision variable is to take.
        active : bool
            Whether the variable is active (``True``) or inactive (``False``).
        flag : str
            Flag to bind to the decision variable.
        """
        super().__init__()
        self.value = value
        self.flag = flag
        self.active = active

    @default_random_state
    def sample(
            self, 
            n: Optional[int] = None,
            random_state=None,
        ) -> Union[object,np.ndarray]:
        """
        Randomly sample ``n`` instances of a decision variable.

        Parameters
        ----------
        n : int, None
            Number of decision variable samples which to draw.
            If ``int``, sample ``n`` decision variables.
            If ``None``, sample a single decision variables.
        
        Returns
        -------
        out : object, np.ndarray
            If ``n`` is ``int``, return a ``np.ndarray`` of shape ``(n,)`` 
            containing sampled decision variables.
            If ``n`` is ``None``, return an ``object`` of a sampled decision 
            variable.
        """
        if n is None:
            return self._sample(1, random_state=random_state)[0]
        else:
            return self._sample(n, random_state=random_state)

    def _sample(
            self, 
            n: int,
            random_state=None,
        ) -> np.ndarray:
        """
        Randomly sample ``n`` instances of a decision variable.
        This is an abstract private method governing the behavior of the 
        ``sample`` method.

        Parameters
        ----------
        n : int
            Number of decision variable samples which to draw.

        Returns
        -------
        out : np.ndarray
            An array of shape ``(n,)`` containing sampled decision variables.
        """
        pass

    def set(
            self, 
            value: object,
        ) -> None:
        """
        Set the value of a decision variable.

        Parameters
        ----------
        value : object
            Value to assign to the decision variable.
        """
        self.value = value

    def get(
            self, 
            **kwargs: Any
        ) -> object:
        """
        Get the value of a decision variable.

        Parameters
        ----------
        kwargs : Any
            Additional keyword arguments.
        
        Returns
        -------
        out : object
            The value of the decision variable.
        """
        return self.value


class BoundedVariable(Variable):
    """
    Semi-abstract class for the representation of a bounded decision variable.
    """

    def __init__(
            self, 
            value: Optional[object] = None, 
            bounds: Tuple[Optional[object],Optional[object]] = (None, None), 
            strict: Optional[Tuple[Optional[object],Optional[object]]] = None, 
            **kwargs: Any,
        ) -> None:
        """
        Constructor for the ``BoundedVariable`` class.

        Parameters
        ----------
        value : object
            Value the decision variable is to take.
        bounds : tuple
            A tuple of length 2 containing upper and lower limits for the 
            decision variable.
        strict : tuple, None
            Strict boundaries for the decision variable.
            If ``None``, the value of ``bounds`` is copied to ``strict``.
        kwargs : Any
            Additional keyword arguments for ``active`` and ``flag``.
        """
        # call the Variable constructor 
        super().__init__(value=value, **kwargs)
        self.bounds = bounds

        # if no strict boundaries were provided, consider ``bounds`` as 
        # strict boundaries
        if strict is None:
            strict = bounds
        self.strict = strict

    @property
    def lb(self) -> object:
        """
        Lower bound of the decision variable.

        Returns
        -------
        out : object
            The decision variable lower bound.
        """
        return self.bounds[0]

    @property
    def ub(self) -> object:
        """
        Upper bound of the decision variable.

        Returns
        -------
        out : object
            The decision variable upper bound.
        """
        return self.bounds[1]


class Real(BoundedVariable):
    """
    Class for the representation of bounded, real decision variables.
    """
    # variable type represented by this object class
    vtype = float

    def _sample(
            self, 
            n: int,
            random_state=None,
        ) -> np.ndarray:
        """
        Randomly sample ``n`` instances of a real, bounded decision variable.
        Decision variables are sampled from a uniform distribution.

        This is a private method governing the behavior of the ``sample`` 
        method.

        Parameters
        ----------
        n : int
            Number of decision variable samples which to draw.
        random_state
            Random state for sampling.

        Returns
        -------
        out : np.ndarray
            An array of shape ``(n,)`` containing sampled real, bounded 
            decision variables.
        """
        low, high = self.bounds
        return random_state.uniform(low=low, high=high, size=n)


class Integer(BoundedVariable):
    """
    Class for the representation of bounded, integer decision variables.
    """
    # variable type represented by this object class
    vtype = int

    def _sample(
            self, 
            n: int,
            random_state=None,
        ) -> np.ndarray:
        """
        Randomly sample ``n`` instances of a bounded, integer decision variable.
        Decision variables are sampled from a uniform distribution.

        This is a private method governing the behavior of the ``sample`` 
        method.

        Parameters
        ----------
        n : int
            Number of decision variable samples which to draw.
        random_state
            Random state for sampling.

        Returns
        -------
        out : np.ndarray
            An array of shape ``(n,)`` containing sampled bounded, integer 
            decision variables.
        """
        low, high = self.bounds
        return random_state.integers(low, high + 1, size=n)


class Binary(BoundedVariable):
    """
    Class for the representation of a binary, bounded decision variable.
    """
    # variable type represented by this object class
    vtype = bool

    def _sample(
            self, 
            n: int,
            random_state=None,
        ) -> np.ndarray:
        """
        Randomly sample ``n`` instances of a bounded, binary decision variable.
        Decision variables are sampled from a uniform distribution.

        This is a private method governing the behavior of the ``sample`` 
        method.

        Parameters
        ----------
        n : int
            Number of decision variable samples which to draw.
        random_state
            Random state for sampling.

        Returns
        -------
        out : np.ndarray
            An array of shape ``(n,)`` containing sampled bounded, binary 
            decision variables.
        """
        return random_state.random(size=n) < 0.5


class Choice(Variable):
    """
    Class for the representation of a discrete, subset decision variable.
    """
    # variable type represented by this object class
    vtype = object

    def __init__(
            self, 
            value: Optional[object] = None, 
            options: Optional[ArrayLike] = None, 
            all: Optional[ArrayLike] = None, 
            **kwargs: Any,
        ) -> None:
        """
        Constructor for the ``Choice`` class.

        Parameters
        ----------
        value : object
            Value the decision variable is to take.
        options : ArrayLike, None
            A list of decision variable options from which to choose.
        all : ArrayLike, None
            A strict list of decision variable options from which to choose.
            If ``None``, the value of ``options`` is copied to ``all``.
        kwargs : Any
            Additional keyword arguments for ``active`` and ``flag``.
        """
        # all super constructor
        super().__init__(value=value, **kwargs)
        self.options = options

        # if strict list not provided, set to ``options``
        if all is None:
            all = options
        self.all = all

    def _sample(
            self, 
            n: int,
            random_state=None,
        ) -> np.ndarray:
        """
        Randomly sample ``n`` instances of a discrete, subset decision variable.
        Decision variables are sampled with replacement from a uniform 
        distribution.

        This is a private method governing the behavior of the ``sample`` 
        method.

        Parameters
        ----------
        n : int
            Number of decision variable samples which to draw.
        random_state
            Random state for sampling.

        Returns
        -------
        out : np.ndarray
            An array of shape ``(n,)`` containing sampled bounded, integer 
            decision variables.
        """
        return random_state.choice(self.options, size=n)


def get(
        *args: Tuple[Union[Variable,object],...], 
        size: Optional[Union[tuple,int]] = None, 
        **kwargs: Any
    ) -> Union[tuple,object,None]:
    """
    Get decision variable values from a tuple of ``Variable`` objects.

    Parameters
    ----------
    args : tuple
        A tuple of ``Variable`` or ``object``s.
    size : tuple, int, None
        Size to reshape decision variables.
    kwargs : Any
        Additional keyword arguments to pass to the ``get`` method of the 
        ``Variable`` class when getting decision variable values.
    
    Returns
    -------
    out : tuple, object, None
        Decision variable value(s).
    """
    if len(args) == 0:
        return

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
