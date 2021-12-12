from pyro.poutine.runtime import effectful

from qqn.library.action import action_islegal_eff, all_actions_eff


def policy_prior_default(state):
    # where i left off
    return list(filter(all_actions_eff(), action_islegal_eff))


policy_prior_type = 'policy_prior'
_policy_prior_eff = effectful(policy_prior_default, type=policy_prior_type)


def policy_prior_eff(state):
    args = (state,)
    return _policy_prior_eff(*args)


policy_type = 'policy'
_policy_eff = effectful(policy_prior_eff, type=policy_type)


def policy_eff(state):
    args = (state,)
    return _policy_eff(*args)


def policy_value_eff(policy, state):
    pass
