from copy import copy

from frozendict import frozendict
import pyro
import pyro.infer

from torch import tensor, Tensor
import torch

import pyro.distributions as dist

from qqn.exploration.AbstractAgent import optimize_agent_preferences, print_preferences


def softmax_agent(state, *args, **kwargs):
    next_actions = kwargs.get("next_actions", tensor([0.]))
    number_of_next_actions = len(next_actions)
    action_translation_dict = kwargs.get("action_translation_dict",
                                         frozendict({n: next_actions[n] for n in range(number_of_next_actions)}))
    kwargs["action_translation_dict"] = action_translation_dict

    next_action_weights_func = kwargs.get("next_action_weights_func",
                                          lambda _s, **kw: torch.zeros(number_of_next_actions))
    next_action_weights = next_action_weights_func(state, **kwargs)
    next_action_dist = dist.Categorical(logits=next_action_weights)

    prefix = kwargs.get("prefix")
    suffix = kwargs.get("suffix")
    iteration = kwargs.get("iteration")
    move = kwargs.get("move")
    alpha = kwargs.get("alpha", tensor(1.))
    if not isinstance(alpha, Tensor):
        alpha = tensor(alpha)

    next_action_t = pyro.sample("{}action_{}_{}{}".format(prefix, iteration, move, suffix), next_action_dist)
    next_action_i = action_translation_dict[next_action_t.item()]
    expected_utility = expected_utility_func(state, next_action_i, next_action_dist, **kwargs)

    pyro.factor("{}observation_{}_{}{}".format(prefix, iteration, move, suffix), alpha * expected_utility)
    return next_action_i


def softmax_agent_model(state, *args, **kwargs):
    next_action_weights_func_model = kwargs.get("next_action_weights_func_model")
    is_final_state_func = kwargs.get("is_final_state_func", lambda s, _a: s["time_left"] + s["time_cost"] <= 0)
    transition_func = kwargs.get("transition_func", lambda s, _a: s)
    move = kwargs.get("move", 0)
    kwargs["move"] = move

    is_final_state = False

    while not is_final_state:
        if next_action_weights_func_model is not None:
            next_action = softmax_agent(state, *args, next_action_weights_func=next_action_weights_func_model, **kwargs)
        else:
            next_action = softmax_agent(state, *args, **kwargs)
        is_final_state = is_final_state_func(state, next_action)
        if not is_final_state:
            state = transition_func(state, next_action)
            kwargs["move"] += 1

    return state


def softmax_agent_guide(state, *args, **kwargs):
    next_action_weight_func_guide = kwargs.get("next_action_weights_func_guide")
    is_final_state_func = kwargs.get("is_final_state_func", lambda s, _a: s["time_left"] + s["time_cost"] <= 0)
    transition_func = kwargs.get("transition_func", lambda s, _a: s)
    move = kwargs.get("move", 0)
    kwargs["move"] = move

    is_final_state = False

    while not is_final_state:
        next_action = softmax_agent(state, *args, next_action_weights_func=next_action_weight_func_guide, **kwargs)
        is_final_state = is_final_state_func(state, next_action)
        if not is_final_state:
            state = transition_func(state, next_action)
            kwargs["move"] += 1

    return state


def expected_utility_func(*args, **kwargs):
    state = kwargs.get("state", args[0])
    action = kwargs.get("action", args[1])

    final_utility_func = kwargs.get("final_utility_func", lambda _s, _a: tensor(0.0))
    util = final_utility_func(state, action)

    depth = kwargs.pop("depth", 0)
    max_depth = kwargs.get("max_depth", None)
    is_final_state_func = kwargs.get("is_final_state", lambda s, _a: s["time_left"] + s["time_cost"] <= 0)

    if is_final_state_func(state, action) or (max_depth is not None and depth >= max_depth):
        return util

    transition_func = kwargs.get("transition_func", lambda s, _a: s)
    next_state = transition_func(state, action)

    next_action_weights_func = kwargs.get("next_action_weights_func", args[2])
    next_action_weights = next_action_weights_func(next_state, **kwargs)
    action_dist = dist.Categorical(logits=next_action_weights)

    prefix = kwargs.get("prefix")
    suffix = kwargs.get("suffix")

    num_samples = kwargs.get("num_samples", int(max(len(kwargs.get("next_actions")) * 2,
                                                    10 ** ((5 if max_depth is None else max_depth) - depth
                                                           ))))
    samples = []
    action_translation_dict = kwargs.get("action_translation_dict")
    branch = kwargs.pop("branch", [])
    for i in pyro.plate(
            '{}expected_utility_{}_{}_sample_{}_{}{}'.format(prefix, kwargs["iteration"], kwargs["move"], depth, branch,
                                                             suffix),
            num_samples):
        next_action_t = pyro.sample(
            "{}expected_utility_{}_{}_sample_{}_{}{}_{}".format(prefix, kwargs["iteration"], kwargs["move"], depth,
                                                                branch, suffix, i),
            action_dist)
        next_action_i = action_translation_dict[next_action_t.item()]
        this_branch = copy(branch)
        this_branch.append(i)
        samples.append(
            expected_utility_func(next_state, next_action_i, next_action_weights_func, depth=depth + 1,
                                  branch=this_branch,
                                  **kwargs))
    e_dist = tensor(samples).float().cpu().mean()
    return util + e_dist


########################################################################################################################
# Example (Left, Stay, Right), start: 0, goal: 3
########################################################################################################################

def example_next_action_weights_func_model(state, **kwargs):
    return torch.zeros(3)


def example_next_action_weights_func_guide(state, **kwargs):
    value = state["value"]
    prefix = kwargs.get("prefix")
    suffix = kwargs.get("suffix")
    param_name = "{}preferences_for_{}{}".format(prefix, value, suffix)
    if param_name not in pyro.get_param_store():
        pyro.param(param_name, example_next_action_weights_func_model(state))
    return pyro.get_param_store()[param_name]


def example_is_final_state_func(state, action):
    return state["time_left"] + state["time_cost"] <= 0


def example_final_utility_func(state, action):
    u = tensor(-1.)
    if "value" in state:
        assert not isinstance(state["value"], Tensor)
        u = tensor(0.)
        if state["value"] + action == 3:
            u = tensor(1.)
    return u


def example_transition_func(state, action):
    new_state = {"value": 0, "time_left": 1} | copy(state)
    new_state["value"] += action
    new_state["time_left"] += new_state["time_cost"]
    return new_state


optimize_agent_preferences({"value": 0, "time_left": 4, "time_cost": -1},
                           model=softmax_agent_model,
                           guide=softmax_agent_guide,
                           prefix="",
                           suffix="_mdp",
                           is_final_state_func=example_is_final_state_func,
                           transition_func=example_transition_func,
                           final_utility_func=example_final_utility_func,
                           next_action_weights_func_model=example_next_action_weights_func_model,
                           next_action_weights_func_guide=example_next_action_weights_func_guide,
                           next_actions=[-1, 0, 1],
                           num_samples=6,
                           opt_steps=1000,
                           alpha=1000.,
                           display_preferences=print_preferences,
                           opt_progress=False
                           )

########################################################################################################################
# Example Gridworld
########################################################################################################################

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

# optimize_agent_preferences({
#     "grid": grid, "position": (3, 1), "time_left": 9, "time_cost": -0.1
# },
#     model=softmax_agent_model,
#     guide=softmax_agent_guide,
#     final_utility_func=grid_utility_func,
#     transition_func=grid_transition_func,
#     next_actions=["N", "O", "S", "W"]
# )
