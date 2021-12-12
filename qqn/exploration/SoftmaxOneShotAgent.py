import pyro
import pyro.distributions as dist

import torch
from torch import tensor, Tensor


def max_agent_model(state, obs=None, *args, **kwargs):
    next_actions = kwargs.pop("next_actions", tensor([0.]))
    next_action_dist = kwargs.pop("next_action_dist",
                                  dist.Categorical(torch.ones(len(next_actions)) / len(next_actions)))
    transition = kwargs.pop("transition", lambda s, _a: s)
    action_translation = kwargs.pop("action_translation", lambda a: a)
    state_translation = kwargs.pop("state_translation", lambda s: s)

    next_action: Tensor = pyro.sample("action", next_action_dist)
    next_action_ = next_action.cpu().apply_(action_translation)
    next_state = next_action_.apply_(lambda a: state_translation(transition(state, next_action_)))


def softmax_agent_model(state, observation=None, **kwargs):
    next_actions = kwargs.pop("next_actions", tensor([0.]))
    next_action_dist = kwargs.pop("next_action_dist",
                                  dist.Categorical(torch.ones(len(next_actions)) / len(next_actions)))
    transition = kwargs.pop("transition", lambda s, _a: s)
    action_translation = kwargs.pop("action_translation", lambda a: a)
    state_translation = kwargs.pop("state_translation", lambda s: s)
    utility_func = kwargs.pop("utility_func", None)
    utility_dict = kwargs.pop("utility_dict", None)
    alpha = kwargs.pop("alpha", tensor(1.0))

    if utility_dict is None and utility_func is None:
        utility_func = lambda s: tensor(0.0)

    next_action: Tensor = pyro.sample("action", next_action_dist)

    next_state = next_action.cpu().apply_(lambda a: state_translation(transition(state, action_translation(a))))
    assert utility_func is not None or utility_dict is not None
    if utility_func is not None:
        utility = next_state.cpu().apply_(utility_func)
    elif utility_dict is not None:
        utility = next_state.cpu().apply_(lambda s: utility_dict.get(s, tensor(0.0)))
    else:
        assert False

    if observation is not None:
        utility.cpu().apply_(lambda u: pyro.factor("observation", alpha * u))


example_utility = {
    "bad": -10,
    "good": 6,
    "spectacular": 8
}

example_action_translation = ['italian', 'french']

def example_transition(state, action):
    next_states = list(example_utility.keys())
    if action == "italian":
        next_probs = [0.2, 0.6, 0.2]
    else:
        next_probs = [0.05, 0.9, 0.05]
    return 
