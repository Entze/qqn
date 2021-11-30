from copy import copy

from frozendict import frozendict
import pyro
import pyro.infer

from torch import tensor
import torch

import pyro.distributions as dist

from qqn.AbstractAgent import global_get_prefix, global_get_suffix, optimize_agent_preferences


def softmax_agent(state, *args, **kwargs):
    next_actions = kwargs.get("next_actions", tensor([0.]))
    number_of_next_actions = len(next_actions)
    action_translation_dict = kwargs.get("action_translation_dict",
                                         frozendict({n: next_actions[n] for n in range(number_of_next_actions)}))
    kwargs["action_translation_dict"] = action_translation_dict

    next_action_dist = kwargs.get("next_action_dist",
                                  dist.Categorical(torch.ones(number_of_next_actions) / float(number_of_next_actions)))

    expected_utility_func = kwargs.get("expected_utility_func", lambda *a, **kw: tensor(0.))
    prefix = kwargs.get("prefix")
    suffix = kwargs.get("suffix")
    alpha = kwargs.get("alpha", tensor(1.))

    next_action_t = pyro.sample("{}action{}".format(prefix, suffix), next_action_dist)
    expected_utility = next_action_t.cpu().apply_(
        lambda a: expected_utility_func(state, action_translation_dict.get(a, a), next_action_dist, **kwargs))

    pyro.factor("{}observation{}".format(prefix, suffix), alpha * expected_utility)
    return next_action_t


def softmax_agent_model(state, *args, **kwargs):
    return softmax_agent(state, *args, **kwargs)


def softmax_agent_guide(state, *args, **kwargs):
    next_actions = kwargs.get("next_actions", tensor([0.]))
    number_of_next_actions = len(next_actions)
    prefix = kwargs.get("prefix")
    suffix = kwargs.get("suffix")
    preferences = pyro.param("{}preferences{}".format(prefix, suffix),
                             torch.ones(number_of_next_actions) / float(number_of_next_actions))

    next_action_dist = dist.Categorical(logits=preferences)

    return softmax_agent(state, *args, next_action_dist=next_action_dist, **kwargs)


def example_final_utility_func(state, action, depth=None):
    u = tensor(-1.)
    if "value" in state:
        if state["value"] + action == 3:
            u = tensor(1.)
        else:
            u = tensor(0.)
    return u


def example_transition_func(state, action):
    new_state = {"value": 0, "time_left": 1} | copy(state)
    new_state["value"] += action
    new_state["time_left"] -= 1
    return new_state


def example_expected_utility_func(*args, **kwargs):
    state = args[0]
    action = args[1]
    depth = kwargs.pop("depth", 0)
    if "time_left" not in state or state["time_left"] is None or state["time_left"] <= 0:
        return example_final_utility_func(state, action)
    action_dist = args[2]
    action_translation_dict = kwargs.get("action_translation_dict")
    prefix = kwargs.get("prefix", global_get_prefix())
    suffix = kwargs.get("suffix", global_get_suffix())
    num_samples = kwargs.get("num_samples", int(10 ** max(0, min(2 - depth, state["time_left"] - 1))))
    current_final_utility = example_final_utility_func(state, action)
    next_state = example_transition_func(state, action)
    next_action = pyro.sample("{}expected_utility_sample_{}{}".format(prefix, depth, suffix), action_dist)
    # depth -= 1
    # utilities = torch.ones(num_samples) * depth
    return current_final_utility + example_expected_utility_func(next_state, next_action, action_dist, depth=depth,
                                                                 **kwargs)


optimize_agent_preferences({"value": 0, "time_left": 3},
                           model=softmax_agent_model,
                           guide=softmax_agent_guide,
                           prefix="",
                           suffix="_mdp",
                           next_actions=[-1, 0, 1],
                           expected_utility_func=example_expected_utility_func,
                           num_samples=1,
                           )
