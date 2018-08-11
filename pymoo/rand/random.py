import numpy as np

from pymoo.configuration import Configuration


# use a singleton class to set the random generator globally
class Singleton:
    __instance = None

    @staticmethod
    def get_instance():
        if Singleton.__instance is None:
            Singleton.__instance = Configuration.rand
            Singleton.__instance.seed(1)
        return Singleton.__instance


def seed(s):
    return Singleton.get_instance().seed(s)


def perm(size):
    return Singleton.get_instance().perm(size).astype(np.int)


def random(*params, size=None):
    if len(params) > 0:
        return Singleton.get_instance().rand(size=params)
    else:
        return Singleton.get_instance().rand(size=size)


def randint(low, high, size=None):
    return Singleton.get_instance().randint(low, high, size=size)
