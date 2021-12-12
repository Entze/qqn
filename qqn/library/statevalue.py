from pyro.poutine.runtime import effectful

from qqn.library.SetValueMessenger import SetValueMessenger
from qqn.library.common import nothing

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
