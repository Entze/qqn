import pyro.distributions as dist
import torch
from pyro.poutine.runtime import effectful
from torch import Tensor

from qqn.library.action import action_generate_eff, action_rate_eff, action_select_eff, action_map_estimate_eff
from qqn.library.action import nr_of_actions_eff
from qqn.library.common import snd, fst


def policy_default(state, *args, **kwargs):
    options = action_generate_eff(state, *args, **kwargs)
    estimations = action_map_estimate_eff(state, options, *args, **kwargs)
    ratings = action_rate_eff(estimations, state, *args, **kwargs)
    return action_select_eff(ratings, state, *args, **kwargs)


policy_type = 'policy'
_policy_eff = effectful(policy_default, type=policy_type)


def policy_eff(state, *args, **kwargs):
    req_args = (state,)
    return _policy_eff(*req_args, *args, **kwargs)


def policy_posterior_default(state, *args, **kwargs):
    actions = action_generate_eff(state, *args, **kwargs)
    estimations = action_map_estimate_eff(actions, state, *args, **kwargs)
    ratings = action_rate_eff(estimations, state, *args, **kwargs)
    if isinstance(ratings, list) and len(ratings) > 0 and isinstance(ratings[0], tuple):
        assert isinstance(ratings, list)
        ratings.sort(key=fst)
        return torch.tensor(list(map(snd, ratings)))
    elif any(isinstance(ratings, t) for t in (list, tuple, Tensor)):
        return torch.as_tensor(ratings)
    elif isinstance(ratings, dist.Categorical):
        return ratings.logits

    return torch.zeros(nr_of_actions_eff())


policy_posterior_type = 'policy_posterior'
_policy_posterior_eff = effectful(policy_posterior_default, type=policy_posterior_type)


def policy_posterior_eff(state, *args, **kwargs):
    req_args = (state,)
    return _policy_posterior_eff(*req_args, *args, **kwargs)


def policy_distribution_default(state):
    return dist.Categorical(logits=policy_posterior_eff(state))


policy_distribution_type = 'policy_posterior'
_policy_distribution_eff = effectful(policy_distribution_default, type=policy_posterior_type)


def policy_distribution_eff(state, *args, **kwargs):
    req_args = (state,)
    return _policy_posterior_eff(*req_args, *args, **kwargs)
