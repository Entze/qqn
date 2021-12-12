from pyro.poutine.runtime import effectful

from qqn.library.common import nothing

_state_embedding_eff = effectful(nothing, type='state_embedding')


def state_embedding_eff(state):
    args = (state,)
    return _state_embedding_eff(*args)


_state_key_eff = effectful(str, type='state_key')


def state_key_eff(state):
    args = (state,)
    return _state_key_eff(*args)


_state_resource_eff()

def state_resource_eff(state):
    args = (state,)
    return _state_resource_eff(*args)
