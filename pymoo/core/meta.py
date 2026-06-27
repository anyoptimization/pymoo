"""Meta-class wrapper for composing classes dynamically."""

from copy import deepcopy


class Meta:
    """Dynamically wrap an object with mixin behavior from a target class."""

    def __init__(self, object, copy=True, clazz=None) -> None:  # noqa: A002
        if clazz is None:
            clazz = self.__class__

        wrapped = object
        if copy:
            wrapped = deepcopy(wrapped)

        self.__class__ = type(  # noqa: E305
            clazz.__name__,
            tuple([clazz] + wrapped.__class__.mro()),
            {},
        )

        self.__dict__ = wrapped.__dict__
        self.__object__ = object
        self.__super__ = wrapped
