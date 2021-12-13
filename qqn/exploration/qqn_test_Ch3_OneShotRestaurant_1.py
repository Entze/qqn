import torch
from torch import tensor

from qqn.library.SetValueMessenger import SetValueMessenger
from qqn.library.SoftmaxAgentMessenger import softmax_agent
from qqn.library.action import nr_of_actions_type
from qqn.library.policy import policy_eff
from qqn.library.simulate import simulate_by_sampling
from qqn.library.state import StateValueFunctionMessenger
from qqn.library.transition import TransitionFunctionMessenger


def concrete_transition_function(state, action):
    return torch.stack([tensor(0), action])


def concrete_state_value_function(state):
    return state[1]


transition = TransitionFunctionMessenger(concrete_transition_function)
state_value = StateValueFunctionMessenger(concrete_state_value_function)
nr_of_actions = SetValueMessenger(nr_of_actions_type, 2)

with transition, state_value, nr_of_actions:
    print(simulate_by_sampling(tensor([1, 0])))
    print(policy_eff(tensor([1, 0])))

with softmax_agent, transition, state_value, nr_of_actions:
    print(simulate_by_sampling(tensor([1, 0])))
    print(policy_eff(tensor([1, 0])))
    print(policy_eff(tensor([1, 0])))
    print(policy_eff(tensor([1, 0])))
