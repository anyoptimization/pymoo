"""Parameter management utilities for accessing and manipulating Variable objects."""

from pymoo.core.variable import Variable


def get_data(obj):
    """Extract dictionary representation of an object.

    Args:
        obj: Object or dictionary to extract data from.

    Returns:
        Dictionary representation of the object or input dict.
    """
    if not isinstance(obj, dict):
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            return {}
    else:
        return obj


def get_params(obj, flag="default", only_active=True):
    """Get parameters from an object using breadth-first search.

    Args:
        obj: Object to extract parameters from.
        flag: Flag to filter parameters by (default: "default").
        only_active: Only include active parameters (default: True).

    Returns:
        Dictionary of parameters nested by object hierarchy.
    """
    return get_params_bfs(obj, flag, only_active)


def get_params_bfs(obj, flag, only_active):
    """Get parameters using breadth-first search.

    Args:
        obj: Object to extract parameters from.
        flag: Flag to filter parameters by.
        only_active: Only include active parameters.

    Returns:
        Dictionary of parameters nested by object hierarchy.
    """
    ret = {}

    q = [(None, obj)]

    visited = set()

    while len(q) > 0:
        prefix, obj = q.pop()

        if isinstance(obj, Variable):
            if obj not in visited and obj.flag == flag and (not only_active or obj.active):
                # find the right spot in the ret dictionary
                e = ret

                for name in prefix[:-1]:
                    if name not in e:
                        e[name] = {}
                    e = e[name]

                # set it so the value
                e[prefix[-1]] = obj

            # add it to have been visited
            visited.add(obj)

        else:
            data = get_data(obj)
            for key in data:
                new_prefix = [key] if prefix is None else prefix + [key]
                entry = (new_prefix, data[key])
                q.append(entry)

    return ret


def get_params_rec(obj, visited, flag, only_active):
    """Recursively get parameters from an object.

    Args:
        obj: Object to extract parameters from.
        visited: Set of already visited Variable objects.
        flag: Flag to filter parameters by.
        only_active: Only include active parameters.

    Returns:
        Dictionary of parameters nested by object hierarchy.
    """
    data = get_data(obj)

    ret = {}
    for k, v in data.items():
        if isinstance(v, Variable):
            if v not in visited and v.flag == flag and (not only_active or v.active):
                ret[k] = v

            visited.add(v)

        else:
            entry = get_params_rec(v, visited, flag, only_active)
            if entry is not None and len(entry) > 0:
                ret[k] = entry
    return ret


def apply_to_params(obj, func_apply):
    """Apply a function to all parameters in an object.

    Args:
        obj: Object whose parameters to modify.
        func_apply: Function to apply to each parameter.
    """
    for _, v in flatten_rec(get_params(obj)):
        func_apply(v)


def deactivate_params(obj):
    """Deactivate all parameters in an object.

    Args:
        obj: Object whose parameters to deactivate.
    """

    def func(param):
        param.active = False

    apply_to_params(obj, func)


def set_params(obj, params, as_value=True):
    """Set parameters on an object.

    Args:
        obj: Object to set parameters on.
        params: Dictionary of parameters to set.
        as_value: If True, call set() method on parameters. If False, assign directly.
    """
    data = get_data(obj)

    for k, v in params.items():
        if isinstance(v, dict):
            set_params(data[k], v)
        else:
            if as_value:
                data[k].set(v)
            else:
                data[k] = v


def flatten(params):
    """Flatten nested parameter dictionary using dot notation for keys.

    Args:
        params: Nested parameter dictionary.

    Returns:
        Flattened dictionary with dot-notation keys.
    """
    return {k: v for k, v in flatten_rec(params)}


def flatten_rec(params, prefix=None):
    """Recursively flatten a nested parameter dictionary.

    Args:
        params: Parameter dict or value to flatten.
        prefix: Current key prefix (built incrementally).

    Yields:
        (key, value) tuples with flattened keys.
    """
    if hasattr(params, "items"):
        for k, v in params.items():
            yield from flatten_rec(v, prefix=f"{prefix}.{k}" if prefix is not None else k)
    else:
        yield prefix, params


def hierarchical(data):
    """Convert flat dot-notation parameter dictionary back to nested form.

    Args:
        data: Flat dictionary with dot-notation keys.

    Returns:
        Nested dictionary with hierarchical structure.
    """
    ret = {}

    groups = {}
    for k, v in data.items():
        a = k.split(".")
        if len(a) > 1:
            prefix = a[0]
            if prefix not in groups:
                groups[prefix] = {}
            groups[prefix][".".join(a[1:])] = v
        else:
            ret[k] = v

    for name, group in groups.items():
        ret[name] = hierarchical(group)

    return ret
