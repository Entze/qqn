def nothing(*args, **kwargs):
    return None


def identity(val):
    return val


def const(*args, **kwargs):
    return args[0]
