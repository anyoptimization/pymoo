import collections


def is_iterable(obj):
    return isinstance(obj, collections.abc.Sequence)


def y_and_n(obj: bool, none='-'):
    if obj is None:
        return none
    else:
        return 'Y' if obj else 'N'


def swap(x, i, j):
    x[i], x[j] = x[j], x[i]
