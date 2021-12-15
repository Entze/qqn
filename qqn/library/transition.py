from pyro.poutine.runtime import effectful

from qqn.library.setvalue_messenger import SetValueMessenger
from qqn.library.common import nothing

transition_type = 'transition'
_transition_eff = effectful(nothing, type=transition_type)


def transition_eff(state, action):
    args = (state, action)
    return _transition_eff(*args)


class BaseTransitionMessenger(SetValueMessenger):

    def __init__(self, authority):
        super().__init__(transition_type, authority)


class TransitionFunctionMessenger(BaseTransitionMessenger):
    def __init__(self, transition_func):
        super().__init__(transition_func)

    def _access(self, *args, **kwargs):
        return self.authority(*args, **kwargs)


class TransitionTableMessenger(BaseTransitionMessenger):
    def __init__(self, transition_table):
        super().__init__(transition_table)

    def _access(self, *args, **kwargs):
        if "state" in kwargs:
            state = kwargs.pop("state", None)
        else:
            if len(args) < 1:
                raise Exception(
                    "Transition table needs 'state' either as keyword-argument or as first element in argument list.")
            state = args[0]
        if "action" in kwargs:
            action = kwargs.pop("action", None)
        else:
            if len(args) < 1:
                raise Exception(
                    "Transition table needs 'action' either as keyword-argument or as second element in argument list.")
            action = args[1]
        return self.authority[state][action]
