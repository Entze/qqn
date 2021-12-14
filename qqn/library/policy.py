import math
from numbers import Number

import torch
from pyro.poutine.runtime import effectful
from torch import Tensor

from qqn.library.action import nr_of_actions_eff
from qqn.library.common import nothing, snd, fst
from qqn.library.option import option_generator_eff, option_rater_eff, option_selector_eff, option_map_estimator_eff, \
    option_collapser_eff
import pyro.distributions as dist


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


def policy_posterior_default(state):
    options = option_generator_eff(state)
    estimations = option_map_estimator_eff(state, options)
    ratings = option_rater_eff(state, estimations)
    if isinstance(ratings, list) and len(ratings) > 0 and isinstance(ratings[0], tuple):
        assert isinstance(ratings, list)
        ratings.sort(key=fst)
        return torch.tensor(list(map(snd, ratings)))
    elif any(isinstance(ratings, t) for t in (list, tuple, Tensor)):
        return torch.as_tensor(ratings)

    return torch.zeros(nr_of_actions_eff())


policy_posterior_type = 'policy_posterior'
_policy_posterior_eff = effectful(policy_posterior_default, type=policy_posterior_type)


def policy_posterior_eff(state):
    args = (state,)
    return _policy_posterior_eff(*args)


def policy_distribution_default(state):
    return dist.Categorical(logits=policy_posterior_eff(state))


policy_distribution_type = 'policy_posterior'
_policy_distribution_eff = effectful(policy_distribution_default, type=policy_posterior_type)


def policy_distribution_eff(state):
    args = (state,)
    return _policy_posterior_eff(*args)
