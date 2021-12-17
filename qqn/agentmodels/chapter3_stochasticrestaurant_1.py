import pyro
import pyro.distributions as dist
import torch
from torch import tensor

from qqn.agentmodels.testsuite import test
from qqn.library.action import nr_of_actions_type
from qqn.library.setvalue_messenger import SetValueMessenger
from qqn.library.state import StateValueFunctionMessenger, state_isfinal_type
from qqn.library.transition import TransitionFunctionMessenger


def concrete_transition_function(state, action):
    if action == 0:
        probs = [0.2, 0.6, 0.2]
    elif action == 1:
        probs = [0.05, 0.9, 0.05]
    else:
        assert False
    outcome = pyro.sample("transition", dist.Categorical(probs=tensor(probs)))
    return torch.stack([tensor(0), outcome])


def concrete_state_value_function(state):
    if state[1] == 0:
        return tensor(-10.)
    elif state[1] == 1:
        return tensor(6.)
    elif state[1] == 2:
        return tensor(8.)
    return tensor(0.)


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
     min_estimation_value=-10,
     max_estimation_value=8,
     alpha=alpha,
     traces=traces,
     progressbar=False)
