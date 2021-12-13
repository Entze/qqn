from pyro.poutine.runtime import effectful

from qqn.library.SetValueMessenger import SetValueMessenger
from qqn.library.common import nothing, fst_default, gt_zero, func_composition, le_zero

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


_resource_left_eff = effectful(le_zero, type='resource_left')


def resource_depleted_eff(resource):
    args = (resource,)
    return _resource_left_eff(*args)


state_isfinal_type = 'state_isfinal'
_state_isfinal_eff = effectful(func_composition(state_resource_eff, resource_depleted_eff), type=state_isfinal_type)


def state_isfinal_eff(state):
    args = (state,)
    return _state_isfinal_eff(*args)


state_value_type = 'state_value'
_state_value_eff = effectful(nothing, type=state_value_type)


def state_value_eff(state):
    args = (state,)
    return _state_value_eff(*args)


class BaseStateValueMessenger(SetValueMessenger):

    def __init__(self, authority):
        super().__init__(state_value_type, authority)


class StateValueFunctionMessenger(BaseStateValueMessenger):
    def __init__(self, state_value_func):
        super().__init__(state_value_func)

    def _access(self, *args, **kwargs):
        return self.authority(*args, **kwargs)
