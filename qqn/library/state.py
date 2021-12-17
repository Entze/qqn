from torch import Tensor

from qqn.library.common import nothing, fst_default, func_composition
from qqn.library.effect import effectful
from qqn.library.setvalue_messenger import SetValueMessenger


def state_key_default(state, *args, **kwargs):
    if isinstance(state, Tensor):
        return str(state)
    elif hasattr(state, "__hash__"):
        try:
            return hash(state)
        except TypeError:
            pass
    return str(state)


state_key_type = 'state_key'
_state_key_eff = effectful(state_key_default, type=state_key_type)


def state_key_eff(state, *args, **kwargs):
    req_args = (state,)
    return _state_key_eff(*req_args, *args, **kwargs)


def state_resource_default(state, *args, **kwargs):
    return fst_default(state)


state_resource_type = 'state_resource'
_state_resource_eff = effectful(state_resource_default, type=state_resource_type)


def state_resource_eff(state, *args, **kwargs):
    req_args = (state,)
    return _state_resource_eff(*req_args, *args, **kwargs)


def resource_depleted_default(resource):
    return resource <= 0


resource_depleted_type = 'resource_depleted'
_resource_depleted_eff = effectful(resource_depleted_default, type=resource_depleted_type)


def resource_depleted_eff(resource, *args, **kwargs):
    req_args = (resource,)
    return _resource_depleted_eff(*req_args, *args, **kwargs)


state_isfinal_type = 'state_isfinal'
_state_isfinal_eff = effectful(func_composition(state_resource_eff, resource_depleted_eff), type=state_isfinal_type)


def state_isfinal_eff(state, *args, **kwargs):
    req_args = (state,)
    return _state_isfinal_eff(*req_args, *args, **kwargs)


state_value_type = 'state_value'
_state_value_eff = effectful(nothing, type=state_value_type)


def state_value_eff(state, *args, **kwargs):
    req_args = (state,)
    return _state_value_eff(*req_args, *args, **kwargs)


def update_belief_default(state):
    return state


update_belief_type = 'update_belief'
_update_belief_eff = effectful(update_belief_default, type=update_belief_type)


def update_belief_eff(state, *args, **kwargs):
    req_args = (state,)
    return _update_belief_eff(*req_args, *args, **kwargs)


class BaseStateValueMessenger(SetValueMessenger):

    def __init__(self, authority):
        super().__init__(state_value_type, authority)


class StateValueFunctionMessenger(BaseStateValueMessenger):
    def __init__(self, state_value_func):
        super().__init__(state_value_func)

    def _access(self, *args, **kwargs):
        return self.authority(*args, **kwargs)
