import torch
from torch import tensor

import qqn.initial_exploration.gridworld as gw
from qqn.agentmodels.testsuite import test
from qqn.library.action import nr_of_actions_type, action_islegal_type
from qqn.library.setvalue_messenger import SetValueMessenger
from qqn.library.state import StateValueFunctionMessenger, state_isfinal_type
from qqn.library.transition import TransitionFunctionMessenger


def concrete_transition_function(state, action):
    assert concrete_action_islegal_function(action, state)
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
    new_state[1:] += tensor(t)
    new_state[0] -= 1
    return new_state


donut_s = tensor(1.)
donut_n = tensor(1.)
veg = tensor(3.)
noodle = tensor(2.)

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


def concrete_state_value_function(state):
    timeleft, x, y = state[0], state[1], state[2]
    timeused = initial_state[0] - timeleft
    cost = timeused * 0.1
    if x == 0 and y == 7:
        return donut_s - cost
    elif x == 2 and y == 2:
        return donut_n - cost
    elif x == 4 and y == 0:
        return veg - cost
    elif x == 5 and y == 5:
        return noodle - cost
    return tensor(0.) - cost


def concrete_action_islegal_function(action, state):
    legal_actions = gw.allowed_actions(grid_t, state)
    return legal_actions[action]


def concrete_state_isfinal_function(state):
    return state[0] <= 0 or concrete_state_value_function(state) > 0


transition = TransitionFunctionMessenger(concrete_transition_function)
state_value = StateValueFunctionMessenger(concrete_state_value_function)
action_islegal = SetValueMessenger(action_islegal_type, concrete_action_islegal_function)
state_isfinal = SetValueMessenger(state_isfinal_type, concrete_state_isfinal_function)
nr_of_actions = SetValueMessenger(nr_of_actions_type, 4)

initial_state = tensor([9, 3, 6])
alpha = 30.
traces = 100

test(initial_state=initial_state,
     nr_of_actions=nr_of_actions,
     transition=transition,
     state_value=state_value,
     action_islegal=action_islegal,
     state_isfinal=state_isfinal,
     min_estimation_value=-0.9,
     max_estimation_value=3,
     alpha=alpha,
     traces=traces,
     progressbar=False)
