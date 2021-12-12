from pyro.poutine.runtime import effectful

from qqn.library.common import nothing

_action_value_eff = effectful(nothing, type='action_value')

def action_value_eff(state, action):
    pass