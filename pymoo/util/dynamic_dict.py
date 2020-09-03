
class DynamicDict(dict):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super(DynamicDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delitem__(self, key):
        super(DynamicDict, self).__delitem__(key)
        del self.__dict__[key]

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __delattr__(self, item):
        self.__delitem__(item)

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self
