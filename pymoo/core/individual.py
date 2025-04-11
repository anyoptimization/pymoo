"""
Module containing infrastructure for representing individuals in 
population-based optimization algorithms.
"""

# public API for when using ``from pymoo.core.individual import *``
__all__ = [
    "default_config",
    "Individual",
    "calc_cv",
    "constr_to_cv",
]

import copy
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union
from warnings import warn
import numpy as np


def default_config() -> dict:
    """
    Get default constraint violation configuration settings.

    Returns
    -------
    out : dict
        A dictionary of default constraint violation settings.
    """
    return dict(
        cache = True,
        cv_eps = 0.0,
        cv_ieq = dict(scale=None, eps=0.0, pow=None, func=np.sum),
        cv_eq = dict(scale=None, eps=1e-4, pow=None, func=np.sum),
    )


class Individual:
    """
    Base class for representing an individual in a population-based 
    optimization algorithm.
    """

    # function: function to generate default configuration settings
    default_config = default_config

    def __init__(
            self, 
            config: Optional[dict] = None, 
            **kwargs: Any,
        ) -> None:
        """
        Constructor for the ``Invididual`` class.

        Parameters
        ----------
        config : dict, None
            A dictionary of configuration metadata.
            If ``None``, use a class-dependent default configuration.
        kwargs : Any
            Additional keyword arguments containing data which is to be stored 
            in the ``Individual``.
        """
        # set decision variable vector to None
        self._X = None

        # set values objective(s), inequality constraint(s), equality 
        # contstraint(s) to None
        self._F = None
        self._G = None
        self._H = None
        
        # set first derivatives of objective(s), inequality constraint(s), 
        # equality contstraint(s) to None
        self._dF = None
        self._dG = None
        self._dH = None
        
        # set second derivatives of objective(s), inequality constraint(s), 
        # equality contstraint(s) to None
        self._ddF = None
        self._ddG = None
        self._ddH = None

        # set constraint violation value to None
        self._CV = None

        self.evaluated = None

        # initialize all the local variables
        self.reset()

        # a local storage for data
        self.data = {}

        # the config for this individual
        if config is None:
            config = Individual.default_config()
        self.config = config

        for k, v in kwargs.items():
            if k in self.__dict__:
                self.__dict__[k] = v
            elif "_" + k in self.__dict__:
                self.__dict__["_" + k] = v
            else:
                self.data[k] = v

    def reset(
            self, 
            data: bool = True,
        ) -> None:
        """
        Reset the value of objective(s), inequality constraint(s), equality 
        constraint(s), their first and second derivatives, the constraint 
        violation, and the metadata to empty values.

        Parameters
        ----------
        data : bool
            Whether to reset metadata associated with the ``Individiual``.
        """
        # create an empty array to share
        empty = np.array([])

        # design variables
        self._X = empty

        # objectives and constraint values
        self._F = empty
        self._G = empty
        self._H = empty

        # first order derivation
        self._dF = empty
        self._dG = empty
        self._dH = empty

        # second order derivation
        self._ddF = empty
        self._ddG = empty
        self._ddH = empty

        # if the constraint violation value to be used
        self._CV = None

        if data:
            self.data = {}

        # a set storing what has been evaluated
        self.evaluated = set()

    def has(
            self, 
            key: str,
        ) -> bool:
        """
        Determine whether an individual has a provided key or not.

        Parameters
        ----------
        key : str
            The key for which to test.
        
        Returns
        -------
        out : bool
            Whether the ``Individual`` has the provided key.
        """
        return hasattr(self.__class__, key) or key in self.data

    # -------------------------------------------------------
    # Values
    # -------------------------------------------------------

    @property
    def X(self) -> np.ndarray:
        """
        Get the decision vector for an individual.

        Returns
        -------
        out : np.ndarray
            The decision variable for the individual.
        """
        return self._X

    @X.setter
    def X(self, value: np.ndarray) -> None:
        """
        Set the decision vector for an individual.

        Parameters
        ----------
        value : np.ndarray
            The decision variable for the individual.
        """
        self._X = value

    @property
    def F(self) -> np.ndarray:
        """
        Get the objective function vector for an individual.

        Returns
        -------
        out : np.ndarray
            The objective function vector for the individual.
        """
        return self._F

    @F.setter
    def F(self, value: np.ndarray) -> None:
        """
        Set the objective function vector for an individual.

        Parameters
        ----------
        value : np.ndarray
            The objective function vector for the individual.
        """
        self._F = value

    @property
    def G(self) -> np.ndarray:
        """
        Get the inequality constraint vector for an individual.

        Returns
        -------
        out : np.ndarray
            The inequality constraint vector for the individual.
        """
        return self._G

    @G.setter
    def G(self, value: np.ndarray) -> None:
        """
        Set the inequality constraint vector for an individual.

        Parameters
        ----------
        value : np.ndarray
            The inequality constraint vector for the individual.
        """
        self._G = value

    @property
    def H(self) -> np.ndarray:
        """
        Get the equality constraint vector for an individual.

        Returns
        -------
        out : np.ndarray
            The equality constraint vector for the individual.
        """
        return self._H

    @H.setter
    def H(self, value: np.ndarray) -> None:
        """
        Get the equality constraint vector for an individual.

        Parameters
        ----------
        value : np.ndarray
            The equality constraint vector for the individual.
        """
        self._H = value

    @property
    def CV(self) -> np.ndarray:
        """
        Get the constraint violation vector for an individual by either reading 
        it from the cache or calculating it.

        Returns
        -------
        out : np.ndarray
            The constraint violation vector for an individual.
        """
        config = self.config
        cache = config["cache"]

        if cache and self._CV is not None:
            return self._CV
        else:
            self._CV = np.array([calc_cv(G=self.G, H=self.H, config=config)])
            return self._CV

    @CV.setter
    def CV(self, value: np.ndarray) -> None:
        """
        Set the constraint violation vector for an individual.

        Parameters
        ----------
        value : np.ndarray
            The constraint violation vector for the individual.
        """
        self._CV = value

    @property
    def FEAS(self) -> np.ndarray:
        """
        Get whether an individual is feasible for each constraint.

        Returns
        -------
        out : np.ndarray
            An array containing whether each constraint is feasible for an 
            individual.
        """
        eps = self.config.get("cv_eps", 0.0)
        return self.CV <= eps

    # -------------------------------------------------------
    # Gradients
    # -------------------------------------------------------

    @property
    def dF(self) -> np.ndarray:
        """
        Get the objective function vector first derivatives for an individual.

        Returns
        -------
        out : np.ndarray
            The objective function vector first derivatives for the individual.
        """
        return self._dF

    @dF.setter
    def dF(self, value: np.ndarray) -> None:
        """
        Set the objective function vector first derivatives for an individual.

        Parameters
        ----------
        value : np.ndarray
            The objective function vector first derivatives for the individual.
        """
        self._dF = value

    @property
    def dG(self) -> np.ndarray:
        """
        Get the inequality constraint(s) first derivatives for an individual.

        Returns
        -------
        out : np.ndarray
            The inequality constraint(s) first derivatives for the individual.
        """
        return self._dG

    @dG.setter
    def dG(self, value: np.ndarray) -> None:
        """
        Set the inequality constraint(s) first derivatives for an individual.

        Parameters
        ----------
        value : np.ndarray
            The inequality constraint(s) first derivatives for the individual.
        """
        self._dG = value

    @property
    def dH(self) -> np.ndarray:
        """
        Get the equality constraint(s) first derivatives for an individual.

        Returns
        -------
        out : np.ndarray
            The equality constraint(s) first derivatives for the individual.
        """
        return self._dH

    @dH.setter
    def dH(self, value: np.ndarray) -> None:
        """
        Set the equality constraint(s) first derivatives for an individual.

        Parameters
        ----------
        value : np.ndarray
            The equality constraint(s) first derivatives for the individual.
        """
        self._dH = value

    # -------------------------------------------------------
    # Hessians
    # -------------------------------------------------------

    @property
    def ddF(self) -> np.ndarray:
        """
        Get the objective function vector second derivatives for an individual.

        Returns
        -------
        out : np.ndarray
            The objective function vector second derivatives for the individual.
        """
        return self._ddF

    @ddF.setter
    def ddF(self, value: np.ndarray) -> None:
        """
        Set the objective function vector second derivatives for an individual.

        Parameters
        ----------
        value : np.ndarray
            The objective function vector second derivatives for the individual.
        """
        self._ddF = value

    @property
    def ddG(self) -> np.ndarray:
        """
        Get the inequality constraint(s) second derivatives for an individual.

        Returns
        -------
        out : np.ndarray
            The inequality constraint(s) second derivatives for the individual.
        """
        return self._ddG

    @ddG.setter
    def ddG(self, value: np.ndarray) -> None:
        """
        Set the inequality constraint(s) second derivatives for an individual.

        Parameters
        ----------
        value : np.ndarray
            The inequality constraint(s) second derivatives for the individual.
        """
        self._ddG = value

    @property
    def ddH(self) -> np.ndarray:
        """
        Get the equality constraint(s) second derivatives for an individual.

        Returns
        -------
        out : np.ndarray
            The equality constraint(s) second derivatives for the individual.
        """
        return self._ddH

    @ddH.setter
    def ddH(self, value: np.ndarray) -> None:
        """
        Set the equality constraint(s) second derivatives for an individual.

        Parameters
        ----------
        value : np.ndarray
            The equality constraint(s) second derivatives for the individual.
        """
        self._ddH = value

    # -------------------------------------------------------
    # Convenience (value instead of array)
    # -------------------------------------------------------

    @property
    def x(self) -> np.ndarray:
        """
        Convenience property. Get the decision vector for an individual.

        Returns
        -------
        out : np.ndarray
            The decision variable for the individual.
        """
        return self.X

    @property
    def f(self) -> float:
        """
        Convenience property. Get the first objective function value for an individual.

        Returns
        -------
        out : float
            The first objective function value for the individual.
        """
        return self.F[0]

    @property
    def cv(self) -> Union[float,None]:
        """
        Convenience property. Get the first constraint violation value for an 
        individual by either reading it from the cache or calculating it.

        Returns
        -------
        out : float, None
            The constraint violation vector for an individual.
        """
        if self.CV is None:
            return None
        else:
            return self.CV[0]

    @property
    def feas(self) -> bool:
        """
        Convenience property. Get whether an individual is feasible for the 
        first constraint.

        Returns
        -------
        out : bool
            Whether an individual is feasible for the first constraint.
        """
        return self.FEAS[0]

    # -------------------------------------------------------
    # Deprecated
    # -------------------------------------------------------

    @property
    def feasible(self) -> np.ndarray:
        """
        Deprecated. Get whether an individual is feasible for each constraint.

        Returns
        -------
        out : np.ndarray
            An array containing whether each constraint is feasible for an 
            individual.
        """
        warn(
            "The ``feasible`` property for ``pymoo.core.individual.Individual`` is deprecated",
            DeprecationWarning,
            stacklevel = 2,
        )
        return self.FEAS

    # -------------------------------------------------------
    # Other Functions
    # -------------------------------------------------------

    def set_by_dict(
            self, 
            **kwargs: Any
        ) -> None:
        """
        Set an individual's data or metadata using values in a dictionary.

        Parameters
        ----------
        kwargs : Any
            Keyword arguments defining the data to set.
        """
        for k, v in kwargs.items():
            self.set(k, v)

    def set(
            self, 
            key: str, 
            value: object,
        ) -> 'Individual':
        """
        Set an individual's data or metadata based on a key and value.

        Parameters
        ----------
        key : str
            Key of the data for which to set.
        value : object
            Value of the data for which to set.
        
        Returns
        -------
        out : Individual
            A reference to the ``Individual`` for which values were set.
        """
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.data[key] = value
        return self

    def get(
            self, 
            *keys: str,
        ) -> Union[tuple,object]:
        """
        Get the values for one or more keys for an individual.

        Parameters
        ----------
        keys : str
            Keys for which to get values.

        Returns
        -------
        out : tuple, object
            If more than one key provided, return a ``tuple`` of retrieved values.
            If a single key provided, return the retrieved value.
        """
        ret = []

        for key in keys:
            if hasattr(self, key):
                v = getattr(self, key)
            elif key in self.data:
                v = self.data[key]
            else:
                v = None

            ret.append(v)

        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)

    def duplicate(
            self, 
            key: str, 
            new_key: str,
        ) -> None:
        """
        Duplicate a key to a new key.

        Parameters
        ----------
        key : str
            Name of the key to duplicated.
        new_key : str
            Name of the key to which to duplicate the original key.
        """
        self.set(new_key, self.get(key))

    def new(self) -> 'Individual':
        """
        Create a new instance of this class.

        Returns
        -------
        out : Individual
            A new instance of an ``Individual``.
        """
        return self.__class__()

    def copy(
            self, 
            other: Optional['Individual'] = None, 
            deep: bool = True,
        ) -> 'Individual':
        """
        Copy an individual.

        Parameters
        ----------
        other : Individual, None
            The individual to copy. If ``None``, assumed to be self.
        deep : bool
            Whether to deep copy the individual.
        
        Returns
        -------
        out : Individual
            A copy of the individual.
        """
        obj = self.new()

        # if not provided just copy yourself
        if other is None:
            other = self

        # the data the new object needs to have
        D = other.__dict__

        # if it should be a deep copy do it
        if deep:
            D = copy.deepcopy(D)

        for k, v in D.items():
            obj.__dict__[k] = v

        return obj


def calc_cv(
        G: Optional[np.ndarray] = None, 
        H: Optional[np.ndarray] = None, 
        config: Optional[dict] = None,
    ) -> np.ndarray:
    """
    Calculate the constraint violation(s) for a set of inequality constraint(s), 
    equality constraint(s), and a scoring configuration.

    Parameters
    ----------
    G : np.ndarray, None
        A vector of inequality constraint(s).
    H : np.ndarray, None
        A vector of equality constraint(s).
    config : dict, None
        A dictionary of constraint violation scoring configuration settings.
    
    Returns
    -------
    out : np.ndarray
        An array of constraint violations for each objective.
    """
    if G is None:
        G = np.array([])

    if H is None:
        H = np.array([])

    if config is None:
        config = Individual.default_config()

    if G is None:
        ieq_cv = [0.0]
    elif G.ndim == 1:
        ieq_cv = constr_to_cv(G, **config["cv_ieq"])
    else:
        ieq_cv = [constr_to_cv(g, **config["cv_ieq"]) for g in G]

    if H is None:
        eq_cv = [0.0]
    elif H.ndim == 1:
        eq_cv = constr_to_cv(np.abs(H), **config["cv_eq"])
    else:
        eq_cv = [constr_to_cv(np.abs(h), **config["cv_eq"]) for h in H]

    return np.array(ieq_cv) + np.array(eq_cv)


def constr_to_cv(
        c: Union[np.ndarray,None], 
        eps: float = 0.0, 
        scale: Optional[float] = None, 
        pow: Optional[float] = None, 
        func: object = np.mean,
    ) -> float:
    """
    Convert a constraint to a constraint violation.

    c : np.ndarray
        An array of constraint violations.
    eps : float
        Error tolerance bounds.
    scale : float, None
        The scale to apply to a constraint violation.
        If ``None``, no scale alteration is applied.
    pow : float, None
        A power to apply to a constraint violation.
        If ``None``, no power alteration is applied.
    func : function
        A function to convert multiple constraint violations into a single score.
    """
    if c is None or len(c) == 0:
        return 0.0

    # subtract eps to allow some violation and then zero out all values less than zero
    c = np.maximum(0.0, c - eps)

    # apply init_simplex_scale if necessary
    if scale is not None:
        c = c / scale

    # if a pow factor has been provided
    if pow is not None:
        c = c ** pow

    return func(c)
