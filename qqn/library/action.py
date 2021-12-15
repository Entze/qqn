import math
import random
import statistics
from numbers import Number
from typing import Iterator

import pyro.distributions as dist
import torch
from pyro.distributions import Distribution
from torch import tensor, Tensor

from qqn.library.common import snd, const
from qqn.library.effect import effectful
from qqn.library.state import state_isfinal_eff, state_value_eff
from qqn.library.transition import transition_eff


def all_actions_default(*args, **kwargs):
    return list(range(nr_of_actions_eff(*args, **kwargs)))


all_actions_type = 'all_actions'
_all_actions_eff = effectful(all_actions_default, type=all_actions_type)


def all_actions_eff(*args, **kwargs):
    return _all_actions_eff(*args, **kwargs)


def nr_of_actions_default(*args, **kwargs):
    return len(all_actions_eff(*args, **kwargs))


nr_of_actions_type = 'nr_of_actions'
_nr_of_actions_eff = effectful(nr_of_actions_default, type=nr_of_actions_type)


def nr_of_actions_eff(*args, **kwargs):
    return _nr_of_actions_eff(*args, **kwargs)


action_islegal_type = 'action_islegal'
_action_islegal_eff = effectful(const(True), type=action_islegal_type)


def action_islegal_eff(action, state, *args, **kwargs):
    req_args = (action, state)
    return _action_islegal_eff(*req_args, *args, **kwargs)


def action_prior_default(state, *args, **kwargs):
    nr_of_actions = nr_of_actions_eff(*args, **kwargs)
    logits = torch.zeros(nr_of_actions).float()
    actions = all_actions_eff(*args, **kwargs)

    for a in actions:
        if not action_islegal_eff(state, a):
            logits[a] = float('-inf')
    return logits


action_prior_type = 'action_prior'
_action_prior_eff = effectful(action_prior_default, action_prior_type)


def action_prior_eff(state, *args, **kwargs):
    req_args = (state,)
    return _action_prior_eff(*req_args, *args, **kwargs)


########################################################################################################################
# Generator ############################################################################################################
########################################################################################################################

def action_generate_default(*args, **kwargs):
    return tensor(range(nr_of_actions_eff(*args, **kwargs)))


action_generate_type = 'action_generate'
_action_generate_eff = effectful(action_generate_default, type=action_generate_type)


def action_generate_eff(*args, **kwargs):
    return _action_generate_eff(*args, **kwargs)


########################################################################################################################
# Estimator ############################################################################################################
########################################################################################################################

def action_estimate_default(action, state, depth=0, max_depth=None, *args, **kwargs):
    if not action_islegal_eff(state, action):
        return tensor(float('-inf'))
    next_state = transition_eff(state, action)
    primary = state_value_eff(next_state)
    if state_isfinal_eff(next_state):
        return primary.float()
    if max_depth is not None and depth >= max_depth:
        return primary + action_heuristic_eff(next_state)
    actions = action_generate_eff(next_state)
    estimations = action_map_estimate_eff(actions, next_state, depth + 1, max_depth, *args, **kwargs)
    ratings = action_rate_eff(estimations, next_state, *args, **kwargs)
    secondary = action_collapse_eff(ratings, next_state, *args, **kwargs)
    return (primary + secondary).float()


action_estimate_type = 'action_estimate'
_action_estimate_eff = effectful(action_estimate_default, type=action_estimate_type)


def action_estimate_eff(action, state, *args, **kwargs):
    req_args = (action, state)
    return _action_estimate_eff(*req_args, *args, **kwargs)


########################################################################################################################
# Heuristic ############################################################################################################
########################################################################################################################


action_heuristic_type = 'action_heuristic'
_action_heuristic_eff = effectful(const(tensor(0.)), type=action_heuristic_type)


def action_heuristic_eff(*args, **kwargs):
    return _action_heuristic_eff(*args, **kwargs)


########################################################################################################################
# Map Estimator ########################################################################################################
########################################################################################################################


def action_map_estimate_default(actions, state, *args, **kwargs):
    return torch.stack([action_estimate_eff(action, state, *args, **kwargs) for action in actions])


action_map_estimate_type = 'action_map_estimate'
_action_map_estimate_eff = effectful(action_map_estimate_default, type=action_map_estimate_type)


def action_map_estimate_eff(state, options, *args, **kwargs):
    req_args = (state, options)
    return _action_map_estimate_eff(*req_args, *args, **kwargs)


########################################################################################################################
# Rater ################################################################################################################
########################################################################################################################


def action_rate_default(estimations, *args, **kwargs):
    ratings_unsorted = enumerate(estimations)
    ratings = sorted(ratings_unsorted, key=snd, reverse=True)
    return ratings


action_rate_type = 'action_rate'
_action_rate_eff = effectful(action_rate_default, type=action_rate_type)


def action_rate_eff(estimations, *args, **kwargs):
    req_args = (estimations,)
    return _action_rate_eff(*req_args, *args, **kwargs)


########################################################################################################################
# Selector #############################################################################################################
########################################################################################################################


def action_select_default(ratings, *args, **kwargs):
    if isinstance(ratings, list) and len(ratings) > 0 and isinstance(ratings[0], tuple):
        option, logit = random.choice(ratings)
        while math.isinf(logit):
            option, logit = random.choice(ratings)
        return tensor(option)
    elif isinstance(ratings, Distribution):
        return ratings.sample()
    elif isinstance(ratings, Tensor):
        return dist.Categorical(logits=ratings).sample()
    raise NotImplementedError(
        f"Cannot select from ratings of type {type(ratings).__name__}, you have to write a messenger that processes {str(action_select_type)}")


action_select_type = 'action_select'
_action_select_eff = effectful(action_select_default, type=action_select_type)


def action_select_eff(ratings, *args, **kwargs):
    req_args = (ratings,)
    return _action_select_eff(*req_args, *args, **kwargs)


########################################################################################################################
# Collapser ############################################################################################################
########################################################################################################################


def action_collapse_default(ratings, *args, **kwargs):

    if isinstance(ratings, list) and len(ratings) > 0 and isinstance(ratings[0], tuple):
        if isinstance(ratings[0][1], Number):
            assert isinstance(ratings, list)
            it: Iterator = filter(math.isfinite, ratings)
            return torch.as_tensor(statistics.mean(it))
        elif isinstance(ratings[0][1], Tensor):
            ratings = list(map(snd, ratings))
            ratings = torch.stack(ratings)

    if isinstance(ratings, Tensor):
        ratings = ratings[torch.isfinite(ratings)]
        return torch.mean(ratings)

    raise NotImplementedError(
        f"Cannot select from ratings of type {type(ratings).__name__}, you have to write a messenger that processes {str(action_collapse_type)}")


action_collapse_type = 'action_collapse'
_action_collapse_eff = effectful(action_collapse_default, type=action_collapse_type)


def action_collapse_eff(ratings, *args, **kwargs):
    req_args = (ratings,)
    return _action_collapse_eff(*req_args, *args, **kwargs)
