from copy import deepcopy


class Meta(object):

    def __init__(self, object, copy=True, clazz=None):
        if clazz is None:
            clazz = self.__class__

        wrapped = object
        if copy:
            wrapped = deepcopy(wrapped)

        self.__class__ = type(clazz.__name__,
                              tuple([clazz] + wrapped.__class__.mro()),
                              {})

        self.__dict__ = wrapped.__dict__
        self.__object__ = object
        self.__super__ = wrapped

