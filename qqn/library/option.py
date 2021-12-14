import math
import random
import statistics
from numbers import Number
from typing import Iterable

import torch
from pyro.distributions import Distribution
from pyro.poutine.runtime import effectful
from torch import tensor, Tensor

from qqn.library.action import nr_of_actions_eff, action_islegal_eff
from qqn.library.common import nothing, snd, const
from qqn.library.state import state_isfinal_eff, state_value_eff
from qqn.library.transition import transition_eff


########################################################################################################################
# Generator ############################################################################################################
########################################################################################################################

def option_generator_default(state):
    return tensor(range(nr_of_actions_eff()))


option_generator_type = 'option_generator'
_option_generator_eff = effectful(option_generator_default, type=option_generator_type)


def option_generator_eff(state):
    args = (state,)
    return _option_generator_eff(*args)


########################################################################################################################
# Estimator ############################################################################################################
########################################################################################################################

def option_estimator_default(state, option, depth=0, max_depth=None):
    if not action_islegal_eff(state, option):
        return tensor(float('-inf'))
    next_state = transition_eff(state, option)
    primary = state_value_eff(next_state)
    if state_isfinal_eff(next_state):
        return primary.float()
    if max_depth is not None and depth >= max_depth:
        return primary + option_heuristic_eff(next_state)
    options = option_generator_eff(next_state)
    estimations = option_map_estimator_eff(next_state, options, depth + 1, max_depth)
    ratings = option_rater_eff(next_state, estimations)
    secondary = option_collapser_eff(next_state, ratings)
    return (primary + secondary).float()


option_estimator_type = 'option_estimator'
_option_estimator_eff = effectful(option_estimator_default, type=option_estimator_type)


def option_estimator_eff(state, option, depth=0, max_depth=None):
    args = (state, option, depth, max_depth)
    return _option_estimator_eff(*args)


########################################################################################################################
# Heuristic ############################################################################################################
########################################################################################################################


option_heuristic_type = 'option_heursitic'
_option_heuristic_eff = effectful(const(tensor(0.)), type=option_heuristic_type)


def option_heuristic_eff(*args, **kwargs):
    return _option_heuristic_eff(*args, **kwargs)


########################################################################################################################
# Map Estimator ########################################################################################################
########################################################################################################################


def option_map_estimator_default(state, options, depth=0, max_depth=None):
    return torch.stack([option_estimator_eff(state, option, depth, max_depth) for option in options])


option_map_estimator_type = 'option_map_estimator'
_option_map_estimator_eff = effectful(option_map_estimator_default, type=option_map_estimator_type)


def option_map_estimator_eff(state, options, depth=0, max_depth=None):
    args = (state, options, depth, max_depth)
    return _option_map_estimator_eff(*args)


########################################################################################################################
# Rater ################################################################################################################
########################################################################################################################


def option_rater_default(state, estimations):
    return sorted(enumerate(estimations), key=snd, reverse=True)


option_rater_type = 'option_rater'
_option_rater_eff = effectful(option_rater_default, type=option_rater_type)


def option_rater_eff(state, estimations):
    args = (state, estimations)
    return _option_rater_eff(*args)


########################################################################################################################
# Selector #############################################################################################################
########################################################################################################################


def option_selector_default(state, ratings):
    option, logit = random.choice(ratings)
    while math.isinf(logit):
        option, logit = random.choice(ratings)
    return tensor(option)


option_selector_type = 'option_selector'
_option_selector_eff = effectful(option_selector_default, type=option_selector_type)


def option_selector_eff(state, ratings):
    args = (state, ratings)
    return _option_selector_eff(*args)


########################################################################################################################
# Collapser ############################################################################################################
########################################################################################################################


def option_collapser_default(state, ratings):
    if isinstance(ratings, Tensor):
        ratings = ratings[torch.isfinite(ratings)]
        return torch.mean(ratings)
    elif isinstance(ratings, list) and len(ratings) > 0 and isinstance(ratings[0], Number):
        assert isinstance(ratings, list)
        return torch.as_tensor(statistics.mean(filter(math.isfinite, ratings)))
    return tensor(0.)


option_collapser_type = 'option_collapser'
_option_collapser_eff = effectful(option_collapser_default, type=option_collapser_type)


def option_collapser_eff(state, ratings):
    args = (state, ratings)
    return _option_collapser_eff(*args)
