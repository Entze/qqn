import torch
from torch import tensor, Tensor

from qqn.agentmodels.testsuite import test
from qqn.library.action import nr_of_actions_type
from qqn.library.setvalue_messenger import SetValueMessenger
from qqn.library.state import StateValueFunctionMessenger, state_isfinal_type
from qqn.library.transition import TransitionFunctionMessenger


def concrete_transition_function(state, action):
    if isinstance(action, Tensor):
        return torch.stack([tensor(0), action])
    return tensor([0, action])


def concrete_state_value_function(state):
    return state[1]


oneshot = SetValueMessenger(state_isfinal_type, lambda s: s != 'initial_state')
transition = TransitionFunctionMessenger(concrete_transition_function)
state_value = StateValueFunctionMessenger(concrete_state_value_function)
nr_of_actions = SetValueMessenger(nr_of_actions_type, 2)

initial_state = 'initial_state'
alpha = 1.1
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
