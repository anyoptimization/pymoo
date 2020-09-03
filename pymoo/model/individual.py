import copy

from pymoo.util.dynamic_dict import DynamicDict


class Individual(DynamicDict):

    def has(self, key):
        return key in self

    def set(self, key, val):
        self[key] = val

    def copy(self, deep=False):
        if not deep:
            d = dict(**self.__dict__)
        else:
            d = copy.deepcopy(**self.__dict__)

        ind = Individual(**d)
        return ind

    def get(self, *keys):
        if len(keys) == 1:
            key, = keys
            return super().get(key)
        else:
            return tuple([super().get(key) for key in keys])
