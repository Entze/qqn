import random

import torch
from torch import tensor

from qqn.library.SetValueMessenger import SetValueMessenger
from qqn.library.action import nr_of_actions_type, action_islegal_type
from qqn.library.simulate import simulate_by_sampling
from qqn.library.state import StateValueFunctionMessenger
from qqn.library.transition import TransitionFunctionMessenger

world = tensor([
    [1, 0, 1],
    [0, 0, 0],
    [1, 0, 2]
])


def gen_state(resource, x, y):
    return tensor([resource, x, y])


init_state = gen_state(10, 1, 1)


def concrete_transition_function(state, action):
    next_state = torch.clone(state)
    if action == 0:
        next_state[2] -= 1
    elif action == 1:
        next_state[1] += 1
    elif action == 2:
        next_state[2] += 1
    elif action == 3:
        next_state[1] -= 1
    next_state[0] -= 1
    return next_state


def concrete_state_value_function(state):
    if state[1] == 2 and state[2] == 2:
        return tensor(1.)
    return tensor(0.)


def concrete_action_islegal(state, action):
    permission_mask = allowed_actions(state)
    return permission_mask[action]


def allowed_actions(state):
    x, y = state[1], state[2]
    height = world.size(dim=0)
    width = world.size(dim=1)

    actions = tensor([True for _ in range(4)])
    actions[0] = y != 0
    actions[1] = x != (width - 1)
    actions[2] = y != (height - 1)
    actions[3] = x != 0

    def transition(act):
        return {0: lambda: world[y - 1, x],
                1: lambda: world[y, x + 1],
                2: lambda: world[y + 1, x],
                3: lambda: world[y, x - 1]}[act]

    def validate(act, possible):
        if possible:
            return transition(act)() != 1
        return tensor(False)

    return torch.stack([validate(act, possible) for act, possible in enumerate(actions)])


action_islegal = SetValueMessenger(action_islegal_type, concrete_action_islegal)

transition = TransitionFunctionMessenger(concrete_transition_function)
state_value = StateValueFunctionMessenger(concrete_state_value_function)
nr_of_actions = SetValueMessenger(nr_of_actions_type, 4)

torch.manual_seed(0)
random.seed(0)

with transition, state_value, nr_of_actions, action_islegal:
    print(simulate_by_sampling(init_state))
