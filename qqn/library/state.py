from pyro.poutine.runtime import effectful

from qqn.library.SetValueMessenger import SetValueMessenger
from qqn.library.common import nothing, fst_default, gt_zero, func_composition

_state_embedding_eff = effectful(nothing, type='state_embedding')


def state_embedding_eff(state):
    args = (state,)
    return _state_embedding_eff(*args)


_state_key_eff = effectful(hash, type='state_key')


def state_key_eff(state):
    args = (state,)
    return _state_key_eff(*args)


_state_resource_eff = effectful(fst_default, type='state_resource')


def state_resource_eff(state):
    args = (state,)
    return _state_resource_eff(*args)


_resource_left_eff = effectful(gt_zero, type='resource_left')


def resource_left_eff(resource):
    args = (resource,)
    return _resource_left_eff(*args)


_state_isfinal_eff = effectful(func_composition(resource_left_eff, state_resource_eff), type='state_isfinal')


def state_isfinal_eff(state):
    args = (state,)
    return _state_isfinal_eff(*args)


_state_value_eff = effectful(nothing, type='state_value')


def state_value_eff(state):
    args = (state,)
    return _state_value_eff(*args)


class BaseStateValueMessenger(SetValueMessenger):

    def __init__(self, authority):
        super().__init__('state_value', authority)


class StateValueFunctionMessenger(BaseStateValueMessenger):
    def __init__(self, state_value_func):
        super().__init__(state_value_func)

    def _access(self, *args, **kwargs):
        return self.authority(*args, **kwargs)
