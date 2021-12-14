import torch
from pyro.poutine.runtime import effectful
from torch import tensor

from qqn.library.common import func_composition, const


def all_actions_default():
    return tensor(range(nr_of_actions_eff()))


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
    nr_of_actions = nr_of_actions_eff()  # TODO: BUG
    logits = torch.zeros(nr_of_actions).float()
    actions = all_actions_eff()

    for a in actions:
        if not action_islegal_eff(state, a):
            logits[a] = float('-inf')
    return logits


action_prior_type = 'action_prior'
_action_prior_eff = effectful(action_prior_default, action_prior_type)


def action_prior_eff(state):
    args = (state,)
    return _action_prior_eff(*args)


def action_embedding_eff(action):
    pass
