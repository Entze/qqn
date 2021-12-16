import torch
from torch import tensor
from torch.distributions.utils import logits_to_probs

import qqn.initial_exploration.gridworld as gw
from qqn.library.action import nr_of_actions_type, action_islegal_type, action_estimate_type
from qqn.library.cacher import Cacher
from qqn.library.policy import policy_eff, policy_posterior_eff
from qqn.library.setvalue_messenger import SetValueMessenger
from qqn.library.simulate import simulate_by_sampling
from qqn.library.softmaxagent_messenger import softmax_agent
from qqn.library.state import StateValueFunctionMessenger, state_isfinal_type
from qqn.library.transition import TransitionFunctionMessenger


def concrete_transition_function(state, action):
    assert concrete_action_islegal_function(action, state)
    new_state = torch.clone(state)
    t = [0, 0]
    if action == 0:
        t = [0, -1]
    elif action == 1:
        t = [1, 0]
    elif action == 2:
        t = [0, 1]
    elif action == 3:
        t = [-1, 0]
    new_state[1:] += tensor(t)
    new_state[0] -= 1
    return new_state


west = tensor(1)
east = tensor(10)
hill = tensor(-10)

___ = ' '
H = 'H'
W = 'W'
E = 'V'

grid = [
    [___, ___, ___, ___, ___],
    [___, '#', ___, ___, ___],
    [___, '#', W, '#', E],
    [___, ___, ___, ___, ___],
    [H, H, H, H, H]
]

grid_t = gw.as_tensor(grid)


def concrete_state_value_function(state):
    x, y = state[1], state[2]
    if y > 3:
        return hill
    elif x == 2 and y == 2:
        return west
    elif x == 4 and y == 2:
        return east
    return tensor(0.)


def concrete_action_islegal_function(action, state):
    legal_actions = gw.allowed_actions(grid_t, state)
    return legal_actions[action]


def concrete_state_isfinal_function(state):
    return state[0] <= 0 or concrete_state_value_function(state) > 0


transition = TransitionFunctionMessenger(concrete_transition_function)
state_value = StateValueFunctionMessenger(concrete_state_value_function)
action_islegal = SetValueMessenger(action_islegal_type, concrete_action_islegal_function)
state_isfinal = SetValueMessenger(state_isfinal_type, concrete_state_isfinal_function)
nr_of_actions = SetValueMessenger(nr_of_actions_type, 4)

initial_state = tensor([20, 0, 3])

print("Argmax:")
with nr_of_actions, transition, state_value, state_isfinal, action_islegal, Cacher(types=[action_estimate_type]):
    print(simulate_by_sampling(initial_state))
    print(policy_eff(initial_state))

print('#' * 80)
print("Softmax:")
with nr_of_actions, action_islegal, transition, state_value, state_isfinal, softmax_agent(), Cacher(
        types=[action_estimate_type]):
    print(simulate_by_sampling(initial_state))
    print(policy_eff(initial_state))
    print(logits_to_probs(policy_posterior_eff(initial_state).float()))

# print('#' * 80)
# print("Softmax with sampling:")
# with  nr_of_actions, action_islegal, state_value, state_isfinal, transition, softmax_agent(), SamplingEstimatingAgentMessenger(
#         min_estimation_value=0,
#         max_estimation_value=2,
#         nr_of_bins=21,
#         optimization_steps=2):
#     print(simulate_by_sampling(initial_state))
#     print(policy_eff(initial_state))
#     print(logits_to_probs(policy_posterior_eff(initial_state).float()))

# print('#' * 80)
# print("Cached Softmax with sampling:")
# with action_islegal, nr_of_actions, state_value, state_isfinal, transition, softmax_agent(), SamplingEstimatingAgentMessenger(
#         min_estimation_value=0,
#         max_estimation_value=2,
#         nr_of_bins=21,
#         optimization_steps=100), \
#         Cacher(types=[action_estimate_type]):
#     print(simulate_by_sampling(initial_state))
#     print(policy_eff(initial_state))
#     print(logits_to_probs(policy_posterior_eff(initial_state).float()))
