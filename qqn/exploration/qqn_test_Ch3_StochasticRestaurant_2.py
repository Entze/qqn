import pyro
import torch
from torch import tensor
from torch.distributions.utils import logits_to_probs
import pyro.distributions as dist

from qqn.library.estimator_messenger import SamplingEstimatingAgentMessenger
from qqn.library.setvalue_messenger import SetValueMessenger
from qqn.library.softmaxagent_messenger import softmax_agent
from qqn.library.action import nr_of_actions_type
from qqn.library.policy import policy_eff, policy_posterior_eff
from qqn.library.simulate import simulate_by_sampling
from qqn.library.state import StateValueFunctionMessenger, state_key_type
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


transition = TransitionFunctionMessenger(concrete_transition_function)
state_value = StateValueFunctionMessenger(concrete_state_value_function)
nr_of_actions = SetValueMessenger(nr_of_actions_type, 2)
state_key = SetValueMessenger(state_key_type, str)

# with nr_of_actions, transition, state_value:
#     print(simulate_by_sampling(tensor([1, 0])))
#     print(policy_eff(tensor([1, 0])))
#
# with nr_of_actions, transition, state_value, softmax_agent():
#     print(simulate_by_sampling(tensor([1, 0])))
#     print(policy_eff(tensor([1, 0])))
#     print(logits_to_probs(policy_posterior_eff(tensor([1, 0])).float()))
#
# with nr_of_actions, state_key, state_value, transition, softmax_agent(), \
#         SamplingAgentMessenger(alpha=1., min_state_value=-10, max_state_value=8):
#     print(simulate_by_sampling(tensor([1, 0])))
#     print(policy_eff(tensor([1, 0])))
#     print(logits_to_probs(policy_posterior_eff(tensor([1, 0])).float()))

with nr_of_actions, state_key, state_value, transition, softmax_agent(), SamplingEstimatingAgentMessenger(
        min_estimation_value=-10, max_estimation_value=8, optimization_steps=1_000):
    print(simulate_by_sampling(tensor([1, 0])))
    print(policy_eff(tensor([1, 0])))
    print(logits_to_probs(policy_posterior_eff(tensor([1, 0])).float()))
