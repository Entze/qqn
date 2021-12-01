import time
from collections import defaultdict
from copy import copy

import pyro
import pyro.distributions as dist
from frozendict import frozendict
from pyro.infer import Trace_ELBO, SVI
from pyro.optim import Adam

import torch
from torch import tensor, Tensor
from tqdm import trange

global_prefix = -1
global_suffix = -1


def global_get_prefix():
    global global_prefix
    global_prefix += 1
    return global_prefix


def global_get_suffix():
    global global_suffix
    global_suffix += 1
    return global_suffix


def apply_dict(d, default=None, default_self=True):
    if default_self and default is None:
        return lambda k: d.get(d, k)
    return lambda k: d.get(d, default)


def arg_max_agent_model(state, *args, **kwargs):
    next_actions = kwargs.get("next_actions", tensor([0.]))
    number_of_next_actions = len(next_actions)
    action_translation_dict = kwargs.get("action_translation_dict",
                                         frozendict({n: next_actions[n] for n in range(number_of_next_actions)}))
    next_action_dist = kwargs.get("next_action_dist",
                                  dist.Categorical(torch.ones(number_of_next_actions) / float(number_of_next_actions)))
    next_state_dist_dict = kwargs.get("next_state_dist_dict", frozendict(defaultdict(dist.Unit)))
    utility_dict = kwargs.get("utility_dict", frozendict(defaultdict(lambda: torch.zeros(1))))
    prefix = kwargs.get("prefix")
    suffix = kwargs.get("suffix")

    next_action_t: Tensor = pyro.sample("{}action{}".format(prefix, suffix), next_action_dist)
    next_action_i = next_action_t.item()
    next_action = action_translation_dict.get(next_action_i, next_action_i)
    next_state_dist = next_state_dist_dict.get(next_action, dist.Unit)

    best_state = max(utility_dict, key=utility_dict.get)
    best_state_index = next(i for i, s in enumerate(utility_dict.keys()) if s == best_state)
    next_state = pyro.sample("{}nextstate{}".format(prefix, suffix), next_state_dist, obs=tensor(best_state_index),
                             infer={'is_auxiliary': True})

    return next_action


def arg_max_agent_guide(state, *args, **kwargs):
    next_actions = kwargs.get("next_actions", tensor([0.]))
    number_of_next_actions = len(next_actions)
    action_translation_dict = kwargs.get("action_translation_dict",
                                         frozendict({n: next_actions[n] for n in range(number_of_next_actions)}))
    next_state_dist_dict = kwargs.get("next_state_dist_dict", frozendict(defaultdict(dist.Unit)))
    utility_dict = kwargs.get("utility_dict", frozendict(defaultdict(lambda: torch.zeros(1))))
    prefix = kwargs.get("prefix")
    suffix = kwargs.get("suffix")

    next_action_params = pyro.param("{}preferences{}".format(prefix, suffix), torch.zeros(number_of_next_actions))
    next_action_dist = dist.Categorical(logits=next_action_params)

    next_action_t: Tensor = pyro.sample("{}action{}".format(prefix, suffix), next_action_dist)
    next_action_i = next_action_t.item()
    next_action = action_translation_dict.get(next_action_i, next_action_i)
    next_state_dist = next_state_dist_dict[next_action]

    best_state = max(utility_dict, key=utility_dict.get)
    best_state_index = next(i for i, s in enumerate(utility_dict.keys()) if s == best_state)
    next_state = pyro.sample("{}nextstate{}".format(prefix, suffix), next_state_dist, obs=tensor(best_state_index),
                             infer={'is_auxiliary': True})

    return next_action


def optimize_agent_preferences(state, model, guide, *args, **kwargs):
    optimizer_args = {"lr": 0.025} | kwargs.pop("optimizer_args", {})
    opt_steps = kwargs.pop("opt_steps", None)
    if opt_steps == 0:
        return
    opt_timeout = kwargs.pop("opt_timeout", None)
    opt_progress = kwargs.pop("opt_progress", True)
    display_preferences = kwargs.pop("display_preferences", None)
    next_actions = kwargs.get("next_actions", None)
    if opt_steps is None and opt_timeout is None:
        opt_steps = 100
        opt_timeout = 60
    prefix = kwargs.get("prefix", global_get_prefix())
    kwargs["prefix"] = prefix
    suffix = kwargs.get("suffix", global_get_suffix())
    kwargs["suffix"] = suffix

    optimizer = Adam(optimizer_args)

    svi = SVI(model, guide, optimizer, Trace_ELBO())

    steps = []

    start = time.monotonic()
    progress = trange(opt_steps) if opt_progress else range(opt_steps)
    for i in progress:
        svi.step(state, *args, iteration=i, **kwargs)
        step = {k: pyro.param(k) for k in pyro.get_param_store() if k.startswith(prefix) and k.endswith(suffix)}
        steps.append(step)
        if display_preferences is not None and next_actions is not None:
            display_preferences(step, next_actions, prefix="{}:".format(i))
        if opt_timeout is not None and time.monotonic() - start > opt_timeout:
            break

    if display_preferences is not None and next_actions is not None and steps:
        display_preferences(steps[-1], next_actions)

    return steps


def print_preferences(preferences, actions, decimals=4, prefix=None, outersep='; ', innersep=', '):
    if preferences is None:
        return
    print(
        *(format_preference(name, preference, actions, decimals, innersep) for name, preference in preferences.items()),
        outersep)


def format_preference(name, preference, actions, decimals=4, sep=', '):
    real_prob = preference.exp()
    real_prob_sum = real_prob.sum()
    preferences_dict = {action: real_prob[i] / real_prob_sum for i, action in enumerate(actions)}
    return name + ": " + sep.join(
        map(lambda i: ("{}: {:." + str(decimals) + "f}").format(i[0], i[1]), preferences_dict.items()))


example_next_actions = [
    'italian',
    'french'

]

example_utility_dict = frozendict({
    'pizza': tensor(1.),
    'noodles': tensor(0.)
})

example_next_state_dist_dict = frozendict({
    'italian': dist.Categorical(tensor([0.8, 0.2])),
    'french': dist.Categorical(tensor([0.0, 1.0]))
})

# optimize_agent_preferences('initial_state',
#                            model=arg_max_agent_model,
#                            guide=arg_max_agent_guide,
#                            prefix="restaurant_",
#                            suffix="",
#                            next_actions=example_next_actions,
#                            next_state_dist_dict=example_next_state_dist_dict,
#                            utility_dict=example_utility_dict,
#                            display_preferences=print_preferences,
#                            opt_steps=1000
#                            )
