import math
import random
import statistics
from numbers import Number
from typing import Iterable, Iterator

import pyro.distributions as dist
import torch
from pyro.distributions import Distribution
from pyro.poutine.runtime import effectful
from torch import tensor, Tensor

from qqn.library.common import func_composition
from qqn.library.common import snd, const
from qqn.library.state import state_isfinal_eff, state_value_eff
from qqn.library.transition import transition_eff


def all_actions_default():
    return list(range(nr_of_actions_eff()))


all_actions_type = 'all_actions'
_all_actions_eff = effectful(all_actions_default, type=all_actions_type)


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


def action_prior_default(state):
    nr_of_actions = nr_of_actions_eff()
    logits = torch.zeros(nr_of_actions).float()
    actions = all_actions_eff()

    for a in actions:
        if not action_islegal_eff(state, a):
            logits[a] = float('-inf')
    return logits


action_prior_type = 'action_prior'
_action_prior_eff = effectful(action_prior_default, action_prior_type)


def action_prior_eff(*args, **kwargs):
    return _action_prior_eff(*args, **kwargs)


########################################################################################################################
# Generator ############################################################################################################
########################################################################################################################

def action_generate_default(*args, **kwargs):
    return tensor(range(nr_of_actions_eff()))


action_generate_type = 'option_generator'
_action_generate_eff = effectful(action_generate_default, type=action_generate_type)


def action_generate_eff(*args, **kwargs):
    return _action_generate_eff(*args, **kwargs)


########################################################################################################################
# Estimator ############################################################################################################
########################################################################################################################

def action_estimate_default(state, option, depth=0, max_depth=None, *args, **kwargs):
    if not action_islegal_eff(state, option):
        return tensor(float('-inf'))
    next_state = transition_eff(state, option)
    primary = state_value_eff(next_state)
    if state_isfinal_eff(next_state):
        return primary.float()
    if max_depth is not None and depth >= max_depth:
        return primary + action_heuristic_eff(next_state)
    options = action_generate_eff(next_state)
    estimations = action_map_estimate_eff(next_state, options, depth + 1, max_depth, *args, **kwargs)
    ratings = action_rate_eff(estimations, next_state, *args, **kwargs)
    secondary = action_collapse_eff(ratings, next_state, *args, **kwargs)
    return (primary + secondary).float()


action_estimate_type = 'option_estimator'
_action_estimate_eff = effectful(action_estimate_default, type=action_estimate_type)


def action_estimate_eff(state, option, depth=0, max_depth=None, *args, **kwargs):
    req_args = (state, option, depth, max_depth)
    return _action_estimate_eff(*req_args, *args, **kwargs)


########################################################################################################################
# Heuristic ############################################################################################################
########################################################################################################################


action_heuristic_type = 'option_heursitic'
_action_heuristic_eff = effectful(const(tensor(0.)), type=action_heuristic_type)


def action_heuristic_eff(*args, **kwargs):
    return _action_heuristic_eff(*args, **kwargs)


########################################################################################################################
# Map Estimator ########################################################################################################
########################################################################################################################


def action_map_estimate_default(state, options, depth=0, max_depth=None, *args, **kwargs):
    return torch.stack([action_estimate_eff(state, option, depth, max_depth) for option in options])


action_map_estimate_type = 'option_map_estimator'
_action_map_estimate_eff = effectful(action_map_estimate_default, type=action_map_estimate_type)


def action_map_estimate_eff(state, options, depth=0, max_depth=None, *args, **kwargs):
    req_args = (state, options, depth, max_depth)
    return _action_map_estimate_eff(*req_args, *args, **kwargs)


########################################################################################################################
# Rater ################################################################################################################
########################################################################################################################


def action_rate_default(estimations, *args, **kwargs):
    ratings_unsorted = enumerate(estimations)
    ratings = sorted(ratings_unsorted, key=snd, reverse=True)
    return ratings


action_rate_type = 'option_rater'
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
    return tensor(0)


action_select_type = 'option_selector'
_action_select_eff = effectful(action_select_default, type=action_select_type)


def action_select_eff(ratings, *args, **kwargs):
    req_args = (ratings,)
    return _action_select_eff(*req_args, *args, **kwargs)


########################################################################################################################
# Collapser ############################################################################################################
########################################################################################################################


def action_collapse_default(ratings, *args, **kwargs):
    if isinstance(ratings, Tensor):
        ratings = ratings[torch.isfinite(ratings)]
        return torch.mean(ratings)
    elif isinstance(ratings, list) and len(ratings) > 0 and isinstance(ratings[0], Number):
        assert isinstance(ratings, list)
        it: Iterator = filter(math.isfinite, ratings)
        return torch.as_tensor(statistics.mean(it))
    return tensor(0.)


action_collapse_type = 'option_collapser'
_action_collapse_eff = effectful(action_collapse_default, type=action_collapse_type)


def action_collapse_eff(ratings, *args, **kwargs):
    req_args = (ratings,)
    return _action_collapse_eff(*req_args, *args, **kwargs)
