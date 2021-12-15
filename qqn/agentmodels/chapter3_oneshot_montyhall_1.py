import random

import torch
from torch import tensor
from torch.distributions.utils import logits_to_probs

from qqn.library.action import nr_of_actions_type
from qqn.library.estimator_messenger import SamplingEstimatingAgentMessenger
from qqn.library.policy import policy_eff, policy_posterior_eff
from qqn.library.setvalue_messenger import SetValueMessenger
from qqn.library.simulate import simulate_by_sampling
from qqn.library.softmaxagent_messenger import softmax_agent
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

print("Argmax:")
with oneshot, nr_of_actions, transition, state_value:
    print(simulate_by_sampling('initial_state'))
    print(policy_eff(tensor([1, 0])))

print('#' * 80)
print("Softmax:")
with oneshot, nr_of_actions, transition, state_value, softmax_agent():
    print(simulate_by_sampling('intial_state'))
    print(policy_eff(tensor([1, 0])))
    print(logits_to_probs(policy_posterior_eff(tensor([1, 0])).float()))

print('#' * 80)
print("Softmax with sampling:")
with oneshot, nr_of_actions, state_value, transition, softmax_agent(), SamplingEstimatingAgentMessenger(
        min_estimation_value=0,
        max_estimation_value=1,
        nr_of_bins=4):
    print(simulate_by_sampling('initial_state'))
    print(policy_eff(tensor([1, 0])))
    print(logits_to_probs(policy_posterior_eff(tensor([1, 0])).float()))
