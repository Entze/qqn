import torch
from torch import tensor
from torch.distributions.utils import logits_to_probs

from qqn.library.LearningAgentMessenger import SamplingAgentMessenger
from qqn.library.SetValueMessenger import SetValueMessenger
from qqn.library.SoftmaxAgentMessenger import softmax_agent
from qqn.library.action import nr_of_actions_type
from qqn.library.policy import policy_eff, policy_posterior_eff
from qqn.library.simulate import simulate_by_sampling
from qqn.library.state import StateValueFunctionMessenger
from qqn.library.transition import TransitionFunctionMessenger


def concrete_transition_function(state, action):
    return torch.stack([tensor(0), action])


def concrete_state_value_function(state):
    return state[1]


transition = lambda: TransitionFunctionMessenger(concrete_transition_function)
state_value = lambda: StateValueFunctionMessenger(concrete_state_value_function)
nr_of_actions = lambda: SetValueMessenger(nr_of_actions_type, 2)

with nr_of_actions(), transition(), state_value():
    print(simulate_by_sampling(tensor([1, 0])))
    print(policy_eff(tensor([1, 0])))

with nr_of_actions(), transition(), state_value(), softmax_agent():
    print(simulate_by_sampling(tensor([1, 0])))
    print(policy_eff(tensor([1, 0])))
    print(logits_to_probs(policy_posterior_eff(tensor([1, 0])).float()))

with nr_of_actions(), state_value(), transition(), softmax_agent(), SamplingAgentMessenger():
    print(simulate_by_sampling(tensor([1, 0])))
    print(policy_eff(tensor([1, 0])))
    print(logits_to_probs(policy_posterior_eff(tensor([1, 0])).float()))
