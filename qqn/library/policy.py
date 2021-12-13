from pyro.poutine.runtime import effectful

from qqn.library.option import option_generator_eff, option_rater_eff, option_selector_eff, option_map_estimator_eff


def policy_default(state):
    options = option_generator_eff(state)
    estimations = option_map_estimator_eff(state, options)
    ratings = option_rater_eff(state, estimations)
    return option_selector_eff(state, ratings)


policy_type = 'policy'
_policy_eff = effectful(policy_default, type=policy_type)


def policy_eff(state):
    args = (state,)
    return _policy_eff(*args)
