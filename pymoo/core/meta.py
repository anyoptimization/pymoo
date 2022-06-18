from copy import deepcopy


class Meta(object):

    def __init__(self, object, copy=True):
        wrapped = object
        if copy:
            wrapped = deepcopy(wrapped)

        self.__class__ = type(self.__class__.__name__,
                              tuple([self.__class__] + wrapped.__class__.mro()),
                              {})

        self.__dict__ = wrapped.__dict__
        self.__object__ = object
        self.__super__ = wrapped

