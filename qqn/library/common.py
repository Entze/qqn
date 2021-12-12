from typing import Iterable, Iterator


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
    if hasattr(coll, "__getitem__"):
        coll.get(0, default)
    elif isinstance(coll, Iterator):
        try:
            return next(coll)
        except StopIteration:
            return default
        except Exception:
            raise Exception
    return default


def snd(coll):
    pass


def gt_zero(val):
    return val > 0


def const(*args, **kwargs):
    def ret(*_args, **_kwargs):
        return args[0]

    return ret


def func_composition(f, g):
    def com(val):
        return g(f(val))

    return com
