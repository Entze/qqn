from typing import Iterator


def nothing(*args, **kwargs):
    return None


def identity(val):
    return val


def fst(coll):
    if hasattr(coll, "__getitem__"):
        return coll[0]
    elif isinstance(coll, Iterator):
        return next(coll)
    raise Exception(f"Cannot access first element of type {type(coll).__name__}")


def fst_default(coll, default=None):
    if any(isinstance(coll, t) for t in (list, dict)):
        return coll.get(0, default)
    elif hasattr(coll, "__getitem__"):
        try:
            return coll[0]
        except IndexError:
            return default
    elif isinstance(coll, Iterator):
        return next(coll, default)
    return default


def snd(coll):
    if hasattr(coll, "__getitem__"):
        return coll[1]
    elif isinstance(coll, Iterator):
        next(coll)
        return next(coll)
    raise Exception(f"Cannot access second element of type {type(coll).__name__}")


def gt_zero(val):
    return val > 0


def le_zero(val):
    return val <= 0


def const(val, *args, **kwargs):
    def ret(*_args, **_kwargs):
        return val

    return ret


def func_composition(f, g):
    def com(val):
        return g(f(val))

    return com
