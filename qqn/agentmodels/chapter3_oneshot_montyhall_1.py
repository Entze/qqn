import random

import torch

from qqn.agentmodels.testsuite import test
from qqn.library.action import nr_of_actions_type
from qqn.library.setvalue_messenger import SetValueMessenger
from qqn.library.state import state_isfinal_type, StateValueFunctionMessenger
from qqn.library.transition import TransitionFunctionMessenger


def concrete_transition_function(state, action):
    doors = [1, 2, 3]
    playerdoor = random.choice(doors)
    prizedoor = random.choice(doors)
    montydoors = [door for door in doors if playerdoor != door and prizedoor != door]
    montydoor = random.choice(montydoors)
    switch = action == 1

    return torch.as_tensor(switch) and torch.as_tensor(playerdoor != prizedoor)


def concrete_state_value_function(state):
    return state


oneshot = SetValueMessenger(state_isfinal_type, lambda s: s != 'initial_state')

transition = TransitionFunctionMessenger(concrete_transition_function)
state_value = StateValueFunctionMessenger(concrete_state_value_function)
nr_of_actions = SetValueMessenger(nr_of_actions_type, 2)

initial_state = 'initial_state'
alpha = 5.
traces = 100

test(initial_state=initial_state,
     nr_of_actions=nr_of_actions,
     transition=transition,
     state_value=state_value,
     state_isfinal=oneshot,
     min_estimation_value=0,
     max_estimation_value=1,
     alpha=alpha,
     traces=traces,
     progressbar=False)
