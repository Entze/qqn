import torch
from torch import tensor

import qqn.initial_exploration.gridworld as gw
from qqn.agentmodels.testsuite import test
from qqn.library.action import nr_of_actions_type, action_islegal_type, nr_of_actions_eff
from qqn.library.setvalue_messenger import SetValueMessenger
from qqn.library.state import StateValueFunctionMessenger, state_isfinal_type
from qqn.library.transition import TransitionFunctionMessenger


def concrete_transition_function(state, action):
    assert concrete_action_islegal_function(action, state)
    new_state = torch.clone(state)
    noise = torch.distributions.Bernoulli(0.1)
    missstep = bool(noise.sample())
    if missstep:
        slip_direction = 2 * torch.distributions.Bernoulli(0.5).sample() - 1
        if concrete_action_islegal_function(
                torch.remainder(torch.round(action + slip_direction).int(), nr_of_actions_eff()), state):
            action = torch.remainder(torch.round(action + slip_direction).int(), nr_of_actions_eff())
        elif concrete_action_islegal_function(
                torch.remainder(torch.round(action - slip_direction).int(), nr_of_actions_eff()), state):
            action = torch.remainder(torch.round(action - slip_direction).int(), nr_of_actions_eff())
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


west = tensor(1)
east = tensor(10)
hill = tensor(-10)

___ = ' '
H = 'H'
W = 'W'
E = 'V'

grid = [
    [___, ___, ___, ___, ___],
    [___, '#', ___, ___, ___],
    [___, '#', W, '#', E],
    [___, ___, ___, ___, ___],
    [H, H, H, H, H]
]

grid_t = gw.as_tensor(grid)


def concrete_state_value_function(state):
    time_left = state[0]
    time_passed = initial_state[0] - time_left
    cost = time_passed * 0.1
    x, y = state[1], state[2]
    if y > 3:
        return hill - cost
    elif x == 2 and y == 2:
        return west - cost
    elif x == 4 and y == 2:
        return east - cost
    return tensor(0.) - cost


def concrete_action_islegal_function(action, state):
    legal_actions = gw.allowed_actions(grid_t, state)
    return legal_actions[action]


def concrete_state_isfinal_function(state):
    if state[0] <= 0:
        return True
    x, y = state[1], state[2]
    if y > 3:
        return True
    elif x == 2 and y == 2:
        return True
    return x == 4 and y == 2


transition = TransitionFunctionMessenger(concrete_transition_function)
state_value = StateValueFunctionMessenger(concrete_state_value_function)
action_islegal = SetValueMessenger(action_islegal_type, concrete_action_islegal_function)
state_isfinal = SetValueMessenger(state_isfinal_type, concrete_state_isfinal_function)
nr_of_actions = SetValueMessenger(nr_of_actions_type, 4)

initial_state = tensor([12, 0, 3])
traces = 10
alpha = 10.

test(initial_state=initial_state,
     nr_of_actions=nr_of_actions,
     transition=transition,
     state_value=state_value,
     action_islegal=action_islegal,
     state_isfinal=state_isfinal,
     min_estimation_value=-11.2,
     max_estimation_value=10,
     alpha=alpha,
     traces=traces,
     progressbar=True)
