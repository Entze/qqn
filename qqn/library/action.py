from pyro.poutine.runtime import effectful

from qqn.library.common import nothing, func_composition, const
from qqn.library.policy import policy_eff, policy_value_eff
from qqn.library.state import resource_left_eff, state_resource_eff, state_isfinal_eff, state_value_eff
from qqn.library.transition import transition_eff

all_actions_type = 'all_actions'
_all_actions_eff = effectful(list, type=all_actions_type)


def all_actions_eff():
    args = ()
    return _all_actions_eff(*args)


nr_of_actions_type = 'nr_of_actions'
_nr_of_actions_eff = effectful(func_composition(all_actions_eff, len), type=nr_of_actions_type)


def nr_of_actions_eff():
    args = ()
    return _nr_of_actions_eff(*args)


action_islegal_type = 'action_islegal'
_action_islegal_eff = effectful(const(True), type=action_islegal_type)


def action_islegal_eff(state, action):
    args = (state, action)
    return _action_islegal_eff(*args)


def action_embedding_eff(action):
    pass


action_value_type = 'action_value'
_action_value_eff = effectful(nothing, type=action_value_type)


def action_value_eff(state, action):
    assert not state_isfinal_eff(state), "Action does not a value in a final state."
    next_state = transition_eff(state, action)
    primary = state_value_eff(next_state)
    if state_isfinal_eff(next_state):
        return primary
    policy = policy_eff(next_state)
    secondary = policy_value_eff(policy, next_state)
    return primary + secondary
