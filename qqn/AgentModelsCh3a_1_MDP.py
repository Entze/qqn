from typing import Iterable, List

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.infer import Trace_ELBO, SVI, NUTS, MCMC
from pyro.optim import Adam
from torch import Tensor, tensor
from tqdm import trange

torch.manual_seed(0)
pyro.set_rng_seed(0)

cache = {}


def condition(name: str, ten, knockout=float('-inf')):
    pyro.factor(name, torch.where(ten, 0.0, -100.0))


def index_of(element, iterable: Iterable) -> int:
    for i, e in enumerate(iterable):
        if e == element:
            return i
    return -1


def elem_to_index_tensor(elem, list_embedding: List) -> Tensor:
    return tensor(index_of(elem, list_embedding))


def index_tensor_to_elem(ten: Tensor, list_embedding: List):
    return list_embedding[ten.int().item()]


def un_log_preferences_to_n_preferences(log_preferences_un: Tensor):
    preferences_un = log_preferences_un.exp()
    preferences_un_sum = preferences_un.sum()
    return preferences_un / preferences_un_sum


########################################################################################################


start_state = tensor(0)
total_time = 3


def basic_util(state):
    if state == -3:
        return tensor(100).float()
    return tensor(0).float()


def transition(state, action):
    return state + action


def policy_model(state, time_left):

    next_action = pyro.sample("next_action", dist.Categorical(logits=torch.zeros(3))) - 1
    print(f"@{state.item()} g{next_action.item()}")
    with poutine.block():
        eu = expected_util(state, next_action, time_left)
    pyro.factor("factor", 100 * eu)
    return next_action


def policy(state, time_left):
    if "policy" not in cache:
        cache["policy"] = {}
    state_idx = state.item()
    if state_idx not in cache["policy"]:
        posterior = pyro.infer.Importance(policy_model, num_samples=10).run(state, time_left)
        marginal = pyro.infer.EmpiricalMarginal(posterior)
        cache["policy"][state_idx] = marginal
    return cache["policy"][state_idx]


def eventual_util_model(state, action, time_left):
    next_state = transition(state, action)
    future_action = pyro.sample("future", policy(next_state, time_left)) - 1
    return expected_util(next_state, future_action, time_left)


def eventual_util(state, action, time_left):
    if "eventual_util" not in cache:
        cache["eventual_util"] = {}
    state_idx = state.item()
    if state_idx not in cache["eventual_util"]:
        cache["eventual_util"][state_idx] = {}
    action_idx = action.item()
    if action_idx not in cache["eventual_util"][state_idx]:
        posterior = pyro.infer.Importance(eventual_util_model, num_samples=1000).run(state, action, time_left)
        marginal = pyro.infer.EmpiricalMarginal(posterior)
        cache["eventual_util"][state_idx][action_idx] = marginal
    return cache["eventual_util"][state_idx][action_idx]


def expected_util(state, action, time_left):
    u = basic_util(state)
    new_time_left = time_left - 1
    if new_time_left == 0:
        return u
    return u + eventual_util(state, action, new_time_left).mean


prob_tensor = tensor([policy(start_state, total_time).log_prob(tensor(n)) for n in range(-1,2)])


print(un_log_preferences_to_n_preferences(prob_tensor))
