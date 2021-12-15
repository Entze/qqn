from pyro.poutine.runtime import effectful

from qqn.library.setvalue_messenger import SetValueMessenger
from qqn.library.common import nothing, fst_default, gt_zero, func_composition, le_zero

_state_embedding_eff = effectful(nothing, type='state_embedding')


def state_embedding_eff(state):
    args = (state,)
    return _state_embedding_eff(*args)


state_key_type = 'state_key'
_state_key_eff = effectful(hash, type=state_key_type) # TODO: implement own effect stack


def state_key_eff(state):
    args = (state,)
    return _state_key_eff(*args)


state_resource_type = 'state_resource'
_state_resource_eff = effectful(fst_default, type=state_resource_type)


def state_resource_eff(state):
    args = (state,)
    return _state_resource_eff(*args)


resource_depleted_type = 'resource_depleted'
_resource_depleted_eff = effectful(le_zero, type=resource_depleted_type)


def resource_depleted_eff(resource):
    args = (resource,)
    return _resource_depleted_eff(*args)


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
