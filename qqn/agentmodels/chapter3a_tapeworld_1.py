import torch
from torch import tensor

from qqn.agentmodels.testsuite import test
from qqn.library.action import nr_of_actions_type
from qqn.library.setvalue_messenger import SetValueMessenger
from qqn.library.state import StateValueFunctionMessenger
from qqn.library.transition import TransitionFunctionMessenger


def concrete_transition_function(state, action):
    next_state = torch.clone(state)
    step = action - 1
    next_state[0] -= 1
    next_state[1] += step
    return next_state


def concrete_state_value_function(state):
    if state[1] == 3:
        return tensor(1.)
    return tensor(0.)


transition = TransitionFunctionMessenger(concrete_transition_function)
state_value = StateValueFunctionMessenger(concrete_state_value_function)
nr_of_actions = SetValueMessenger(nr_of_actions_type, 3)

initial_state = tensor([3, 0])

traces = 100

test(initial_state=initial_state,
     nr_of_actions=nr_of_actions,
     transition=transition,
     state_value=state_value,
     min_estimation_value=0,
     max_estimation_value=2,
     traces=traces,
     progressbar=False)
