from typing import Iterable, List

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.infer import Trace_ELBO, SVI, NUTS, MCMC
from pyro.optim import Adam
from torch import Tensor, tensor, float64
from tqdm import trange

torch.manual_seed(0)
pyro.set_rng_seed(0)
torch.autograd.set_detect_anomaly(True)

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
    m = log_preferences_un.max()
    log_preferences_un = log_preferences_un / m
    preferences_un = log_preferences_un.exp()
    preferences_un_sum = preferences_un.sum()
    return preferences_un / preferences_un_sum


########################################################################################################


start_state = tensor(0).float()
total_time = 3

total_bins = 100


def create_bins(min=0, max=1, car=100):
    car_int = int(car)
    width = max - min
    return tensor(list(range(car_int))) / (car_int - 1) * width + min


util_bins = create_bins(car=total_bins)


def nearest_bin(val, min=0, max=1, car=100):
    val = torch.as_tensor(val).float()
    width = max - min
    return torch.minimum(torch.maximum(torch.round(((val - min) / width) * (car - 1)), tensor(0)),
                         torch.as_tensor(car - 1))


def basic_util(state):
    if state == 3:
        return tensor(1).float()
    return tensor(0).float()


def transition(state, action):
    return state + action


def policy_model(state, time_left):
    next_action = (pyro.sample("next_action", dist.Categorical(logits=torch.zeros(3))) - 1).float()
    # print(f"@{state.item()} g{next_action.item()}")
    eu = expected_util(state, next_action, time_left)
    pyro.factor("factor", 100 * eu)
    return next_action


def policy_guide(state, time_left):
    state_idx = state.item()
    preferences = pyro.param(f"policy_{state_idx}_{time_left}_preferences", tensor([2., 1., 1.]))
    next_action = (pyro.sample("next_action", dist.Categorical(logits=preferences)) - 1).float()
    return next_action


def policy(state, time_left):
    if "policy" not in cache:
        cache["policy"] = {}
    state_idx = state.item()
    if state_idx not in cache["policy"]:
        cache["policy"][state_idx] = {}
    if time_left not in cache["policy"][state_idx]:
        svi = SVI(policy_model, policy_guide, Adam({"lr": 0.25}), Trace_ELBO())
        for _ in trange(1_000):
            with poutine.block():
                svi.step(state, time_left)
            cache["policy"][state_idx][time_left] = pyro.param(f"policy_{state_idx}_{time_left}_preferences")

    return cache["policy"][state_idx][time_left]


def eventual_util_model(state, action, time_left):
    next_state = transition(state, action)
    policy_dist = dist.Categorical(logits=policy(next_state, time_left))
    future_action = (pyro.sample("future", policy_dist) - 1).float()
    util = expected_util(next_state, future_action, time_left)
    pyro.sample("eventual_util", dist.Categorical(logits=torch.zeros(util_bins.size(dim=0))),
                obs=nearest_bin(util, 0, 1, total_bins),
                # infer={'is_auxiliary': True}
                )
    return util


def eventual_util_guide(state, action, time_left):
    state_idx = state.item()
    action_idx = action.item()
    next_state = transition(state, action)
    policy_dist = dist.Categorical(logits=policy(next_state, time_left))
    future_action = (pyro.sample("future", policy_dist) - 1).float()
    util = expected_util(next_state, future_action, time_left)
    # print("EU:", util)
    preferences = pyro.param(f"eventual_util_{state_idx}_{action_idx}_{time_left}_preferences",
                             torch.zeros(util_bins.size(dim=0)))
    pyro.sample("eventual_util", dist.Categorical(logits=preferences), obs=nearest_bin(util, 0, 1, total_bins),
                # infer={'is_auxiliary': True}
                )
    return util


def eventual_util(state, action, time_left):
    if "eventual_util" not in cache:
        cache["eventual_util"] = {}
    state_idx = state.item()
    if state_idx not in cache["eventual_util"]:
        cache["eventual_util"][state_idx] = {}
    action_idx = action.item()
    if action_idx not in cache["eventual_util"][state_idx]:
        cache["eventual_util"][state_idx][action_idx] = {}
    if time_left not in cache["eventual_util"][state_idx][action_idx]:
        svi = SVI(eventual_util_model, eventual_util_guide, Adam({"lr": 0.25}), Trace_ELBO())
        for _ in trange(1_000):
            with poutine.block():
                svi.step(state, action, time_left)
            cache["eventual_util"][state_idx][action_idx][time_left] = pyro.param(
                f"eventual_util_{state_idx}_{action_idx}_{time_left}_preferences")

    return cache["eventual_util"][state_idx][action_idx][time_left]


def expected_util(state, action, time_left):
    u = basic_util(state)
    new_time_left = time_left - 1
    if new_time_left == 0:
        return u
    eventual_util_bins = eventual_util(state, action, new_time_left)
    eventual_util_bins_n = un_log_preferences_to_n_preferences(eventual_util_bins)
    #print("UNB:", eventual_util_bins, "NB:", eventual_util_bins_n)
    return u + eventual_util_bins_n.mean()


# print(eventual_util(tensor(2), tensor(1), 2))

for time_left in range(1, total_time):
    for action in range(-1, 2):
        for state in range(-3, 4):
            print()
            eu = expected_util(tensor(state), tensor(action), time_left)
            print("state:", state, "action:", action, "time:", time_left, "eu:", eu.item())
            print()

prob_tensor = policy(start_state, total_time)

print(un_log_preferences_to_n_preferences(prob_tensor))
