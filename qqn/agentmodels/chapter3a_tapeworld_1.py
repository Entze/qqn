import torch
from torch import tensor
from torch.distributions.utils import logits_to_probs

from qqn.library.action import nr_of_actions_type, action_estimate_type
from qqn.library.cacher import Cacher
from qqn.library.estimator_messenger import SamplingEstimatingAgentMessenger
from qqn.library.policy import policy_eff, policy_posterior_eff
from qqn.library.setvalue_messenger import SetValueMessenger
from qqn.library.simulate import simulate_by_sampling
from qqn.library.softmaxagent_messenger import softmax_agent
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

# print("Argmax:")
# with nr_of_actions, transition, state_value:
#     print(simulate_by_sampling(tensor([3, 0])))
#     print(policy_eff(tensor([3, 0])))
#
# print('#' * 80)
# print("Softmax:")
# with nr_of_actions, transition, state_value, softmax_agent():
#     print(simulate_by_sampling(tensor([3, 0])))
#     print(policy_eff(tensor([3, 0])))
#     print(logits_to_probs(policy_posterior_eff(tensor([3, 0])).float()))

print('#' * 80)
print("Softmax with sampling:")
with nr_of_actions, state_value, transition, softmax_agent(), SamplingEstimatingAgentMessenger(
        min_estimation_value=0,
        max_estimation_value=2,
        nr_of_bins=21,
        optimization_steps=2):
    print(simulate_by_sampling(tensor([3, 0])))
    print(policy_eff(tensor([3, 0])))
    print(logits_to_probs(policy_posterior_eff(tensor([3, 0])).float()))

print('#' * 80)
print("Cached Softmax with sampling:")
with nr_of_actions, state_value, transition, softmax_agent(), SamplingEstimatingAgentMessenger(
        min_estimation_value=0,
        max_estimation_value=2,
        nr_of_bins=21,
        optimization_steps=1_000), \
        Cacher(types=[action_estimate_type]):
    print(simulate_by_sampling(tensor([3, 0])))
    print(policy_eff(tensor([3, 0])))
    print(logits_to_probs(policy_posterior_eff(tensor([3, 0])).float()))
