def set_if_none(kwargs, str, val):
    if str not in kwargs:
        kwargs[str] = val

