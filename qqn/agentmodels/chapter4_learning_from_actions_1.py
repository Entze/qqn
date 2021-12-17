import torch
from torch import tensor
from torch.distributions.utils import logits_to_probs
from tqdm import trange

import qqn.initial_exploration.gridworld as gw
from qqn.library.action import action_islegal_type, nr_of_actions_type, action_estimate_type
from qqn.library.block_messenger import block
from qqn.library.cacher import Cacher
from qqn.library.common import snd
from qqn.library.estimator_messenger import sampling_estimating_agent
from qqn.library.max_collapse_messenger import max_collapse_agent
from qqn.library.policy import policy_posterior_eff
from qqn.library.setvalue_messenger import SetValueMessenger
from qqn.library.simulate import simulate_eff
from qqn.library.softmaxagent_messenger import softmax_agent
from qqn.library.state import StateValueFunctionMessenger, state_isfinal_type
from qqn.library.transition import TransitionFunctionMessenger
from qqn.library.weighted_rate_messenger import weighted_rate_agent


def inner_transition_function(state, action):
    assert inner_action_islegal_function(action, state)
    new_state = torch.clone(state)
    t = [0, 0]
    if action == 0:
        t = [0, -1]
    elif action == 1:
        t = [1, 0]
    elif action == 2:
        t = [0, 1]
    elif action == 3:
        t = [-1, 0]
    new_state[1:3] += tensor(t)
    new_state[0] -= 1
    return new_state


___ = ' '
DN = 'DN'
DS = 'DS'
V = 'V'
N = 'N'

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

grid_t = gw.as_tensor(grid)


def inner_state_value_function(state):
    timeleft, x, y = state[0], state[1], state[2]
    timeused = inner_initial_state[0] - timeleft
    donut_n, donut_s, veg, noodle = state[3], state[4], state[5], state[6]
    cost = timeused * 0.008
    if x == 0 and y == 7:
        return donut_s - cost
    elif x == 2 and y == 2:
        return donut_n - cost
    elif x == 4 and y == 0:
        return veg - cost
    elif x == 5 and y == 5:
        return noodle - cost
    return tensor(0.) - cost


def inner_action_islegal_function(action, state):
    legal_actions = gw.allowed_actions(grid_t, state)
    return legal_actions[action]


def inner_state_isfinal_function(state):
    return state[0] <= 0 or inner_state_value_function(state) > 0


inner_transition = TransitionFunctionMessenger(inner_transition_function)
inner_state_value = StateValueFunctionMessenger(inner_state_value_function)
inner_action_islegal = SetValueMessenger(action_islegal_type, inner_action_islegal_function)
inner_state_isfinal = SetValueMessenger(state_isfinal_type, inner_state_isfinal_function)
inner_nr_of_actions = SetValueMessenger(nr_of_actions_type, 4)

inner_initial_state = tensor([11, 3, 6])
alpha = 30.


def favorite_from_traces(traces):
    locations = [
        trace[-1][0] for trace in traces
    ]
    visited = {
        'Donut': sum(1 for location in locations if
                     location[1] == 0 and location[2] == 7 or location[1] == 2 and location[2] == 2),
        'Veg': sum(1 for location in locations if location[1] == 4 and location[2] == 0),
        'Noodle': sum(1 for location in locations if location[1] == 5 and location[2] == 5),
    }
    visited['Nowhere'] = len(traces) - sum(visited.values())
    return sorted(visited.items(), key=snd, reverse=True)[0][0]


donut_fav = tensor([2, 2, 1, 1])
veg_fav = tensor([1, 1, 2, 1])
noodle_fav = tensor([1, 1, 1, 2])


def gen_inner_initial_state(state):
    if state == 0:
        return torch.cat((inner_initial_state, donut_fav))
    elif state == 1:
        return torch.cat((inner_initial_state, veg_fav))
    else:
        return torch.cat((inner_initial_state, noodle_fav))


traces = [
    [(tensor([0, 2, 2]), None)],
    [(tensor([0, 4, 0]), None)],
    [(tensor([0, 4, 0]), None)],
    [(tensor([0, 4, 0]), None)],
    [(tensor([0, 4, 0]), None)],
    [(tensor([0, 4, 0]), None)],
    [(tensor([0, 4, 0]), None)],
    [(tensor([0, 4, 0]), None)],
    [(tensor([0, 4, 0]), None)],
    [(tensor([0, 4, 0]), None)],
]

real_fav = favorite_from_traces(traces)


def outer_state_value_function(state):
    with block():
        with (inner_nr_of_actions,
              inner_transition,
              inner_state_value,
              inner_action_islegal,
              inner_state_isfinal,
              softmax_agent(),
              weighted_rate_agent(alpha=5.),
              max_collapse_agent(),
              Cacher(types=[action_estimate_type])
              ):
            ts = [simulate_eff(gen_inner_initial_state(state)) for _ in trange(100)]
    believed_favorite = favorite_from_traces(ts)
    return tensor(real_fav == believed_favorite, dtype=torch.int)


def outer_transition_function(state, action):
    return action


def outer_state_isfinal_function(state):
    return state != 'initial_state'


outer_transition = TransitionFunctionMessenger(outer_transition_function)
outer_state_value = StateValueFunctionMessenger(outer_state_value_function)
# outer_action_islegal = SetValueMessenger(action_islegal_type, outer_action_islegal_function)
outer_state_isfinal = SetValueMessenger(state_isfinal_type, outer_state_isfinal_function)
outer_nr_of_actions = SetValueMessenger(nr_of_actions_type, 3)

with (outer_nr_of_actions,
      outer_transition,
      outer_state_value,
      outer_state_isfinal,
      softmax_agent(),
      sampling_estimating_agent(optimization_steps=10),
      ):
    print(f"real_fav: {real_fav}")
    # print(simulate_eff('initial_state'))
    print(logits_to_probs(policy_posterior_eff('initial_state')))
