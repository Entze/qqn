import time
from copy import copy

from frozendict import frozendict
import pyro
import pyro.infer

from torch import tensor, Tensor
import torch

import pyro.distributions as dist

from qqn.AbstractAgent import global_get_prefix, global_get_suffix, optimize_agent_preferences, print_preferences


def softmax_agent(state, *args, **kwargs):
    next_actions = kwargs.get("next_actions", tensor([0.]))
    number_of_next_actions = len(next_actions)
    action_translation_dict = kwargs.get("action_translation_dict",
                                         frozendict({n: next_actions[n] for n in range(number_of_next_actions)}))
    kwargs["action_translation_dict"] = action_translation_dict

    next_action_dist = kwargs.get("next_action_dist",
                                  dist.Categorical(torch.ones(number_of_next_actions) / float(number_of_next_actions)))

    prefix = kwargs.get("prefix")
    suffix = kwargs.get("suffix")
    alpha = kwargs.get("alpha", tensor(1.))

    next_action_t = pyro.sample("{}action{}".format(prefix, suffix), next_action_dist)
    next_action_i = action_translation_dict[next_action_t.item()]
    expected_utility = expected_utility_func(state, next_action_i, next_action_dist, **kwargs)

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


def example_final_utility_func(state):
    u = tensor(-1.)
    if "value" in state:
        assert not isinstance(state["value"], Tensor)
        u = tensor(0.)
        if state["value"] == 3:
            u = tensor(1.)
    return u


def example_transition_func(state, action):
    new_state = {"value": 0, "time_left": 1} | copy(state)
    new_state["value"] += action
    new_state["time_left"] += new_state["time_cost"]
    return new_state


def expected_utility_func(*args, **kwargs):
    state = kwargs.get("state", args[0])
    time_left_selector = kwargs.get("time_left_selector", "time_left")
    final_utility_func = kwargs.get("final_utility_func", lambda _s: tensor(0.0))
    depth = kwargs.pop("depth", 0)
    max_depth = kwargs.get("max_depth", None)
    action = kwargs.get("action", args[1])
    transition_func = kwargs.get("transition_func", lambda s, _a: s)
    next_state = transition_func(state, action)
    time_left = next_state[time_left_selector]
    if time_left <= 0 or (max_depth is not None and depth > max_depth):
        return final_utility_func(next_state)
    action_dist = kwargs.get("action_dist", args[2])
    prefix = kwargs.get("prefix")
    suffix = kwargs.get("suffix")
    num_samples = kwargs.get("num_samples", int(max(len(kwargs.get("next_actions")) * 2,
                                                    10 ** min((5 if max_depth is None else max_depth) - depth,
                                                              time_left))))
    samples = []
    action_translation_dict = kwargs.get("action_translation_dict")
    branch = kwargs.pop("branch", [])
    for i in pyro.plate(
            '{}expected_utility_{}_sample_{}_{}_{}{}'.format(prefix, kwargs["iteration"], depth, branch, time_left,
                                                             suffix),
            num_samples):
        next_action_t = pyro.sample(
            "{}expected_utility_{}_sample_{}_{}_{}{}_{}".format(prefix, kwargs["iteration"], depth, branch, time_left,
                                                                suffix, i),
            action_dist)
        next_action_i = action_translation_dict[next_action_t.item()]
        this_branch = copy(branch)
        this_branch.append(i)
        samples.append(
            expected_utility_func(next_state, next_action_i, action_dist, depth=depth + 1, branch=this_branch,
                                  **kwargs))
    return tensor(samples).float().cpu().mean()


optimize_agent_preferences({"value": 0, "time_left": 3, "time_cost": -1},
                           model=softmax_agent_model,
                           guide=softmax_agent_guide,
                           time_left_selector="time_left",
                           prefix="",
                           suffix="_mdp",
                           transition_func=example_transition_func,
                           final_utility_func=example_final_utility_func,
                           next_actions=[-1, 0, 1],
                           opt_steps=0,
                           opt_progress=False
                           )

___ = ' '
DN = {"name": 'Donut N', "utility": tensor(1.)}
DS = {"name": 'Donut S', "utility": tensor(1.)}
V = {"name": 'Veg', "utility": tensor(3.)}
N = {"name": 'Noodle', "utility": tensor(2.)}

grid = [
    ['#', '#', '#', '#', V, '#'],
    ['#', '#', '#', ___, ___, ___],
    ['#', '#', DN, ___, '#', ___],
    ['#', '#', '#', ___, '#', ___],
    ['#', '#', '#', ___, ___, ___],
    ['#', '#', '#', ___, '#', N],
    [___, ___, ___, ___, '#', '#'],
    [DS, '#', '#', ___, '#', '#']
]


def grid_utility_func(state):
    x, y = state["position"]
    block = access_grid(state, x, y)

    if "utility" in block:
        return block["utility"]
    return tensor(0.)


grid_translation = {
    "N": (+0, +1),
    "O": (-1, +0),
    "S": (+0, -1),
    "W": (+1, +0)
}


def access_grid(state, x, y):
    grid = state["grid"]
    max_y = len(grid)
    max_x = len(grid[0])
    outer = max_y - y
    inner = max_x - x
    assert 0 <= outer < max_y and 0 <= inner < max_x, f"Tried to access grid at {inner},{outer}"
    return grid[outer][inner]


def grid_transition_func(state, action):
    x, y = state["position"]
    t_x, t_y = grid_translation[action]
    new_x, new_y = x + t_x, y + t_y
    new_state = {"time_left": 0} | copy(state)
    new_state["time_left"] += state["time_cost"]
    new_state["position"] = new_x, new_y
    return new_state


optimize_agent_preferences({
    "grid": grid, "position": (3, 1), "time_left": 9, "time_cost": -0.1
},
    model=softmax_agent_model,
    guide=softmax_agent_guide,
    final_utility_func=grid_utility_func,
    transition_func=grid_transition_func,
    next_actions=["N", "O", "S", "W"]
)
