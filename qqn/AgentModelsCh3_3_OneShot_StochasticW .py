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


doors = [0, 1, 2]
actions = ['stay', 'switch']

states = ['lost', 'win']

util_table = tensor([0, 1]).float()


def transition_model(state, action: Tensor):
    prize_door: Tensor = pyro.sample("prize", dist.Categorical(logits=torch.zeros(len(doors))))
    alice_door = pyro.sample("alice", dist.Categorical(logits=torch.zeros(len(doors))))
    monty_door_logits = tensor([float(0 if prize_door != door and alice_door != door else '-inf') for door in doors])
    monty_door = pyro.sample("monty", dist.Categorical(logits=monty_door_logits), infer={'is_auxiliary': True})
    won = ((prize_door != alice_door).int() == action).int()
    pyro.sample("won", dist.Categorical(logits=torch.zeros(len(states))), obs=won)
    return won


def transition_guide(state, action: Tensor):
    prize_door: Tensor = pyro.sample("prize", dist.Categorical(logits=torch.zeros(len(doors))))
    alice_door = pyro.sample("alice", dist.Categorical(logits=torch.zeros(len(doors))))
    monty_door_logits = tensor([float(0 if prize_door != door and alice_door != door else '-inf') for door in doors])
    monty_door = pyro.sample("monty", dist.Categorical(logits=monty_door_logits), infer={'is_auxiliary': True})
    won = ((prize_door != alice_door).int() == action).int()
    state_n = index_tensor_to_elem(state, states)
    action_n = index_tensor_to_elem(action, actions)
    preferences = pyro.param(f"transition_preferences_{state_n}_{action_n}", torch.zeros(len(states)))
    pyro.sample("won", dist.Categorical(logits=preferences), obs=won)
    return won


def transition_logits(state, action):
    state_n = index_tensor_to_elem(state, states)
    action_n = index_tensor_to_elem(action, actions)
    if 'transition_logits' not in cache:
        cache['transition_logits'] = {}
    if state_n not in cache['transition_logits']:
        cache['transition_logits'][state_n] = {}
    if action_n not in cache['transition_logits'][state_n]:
        svi = SVI(transition_model, transition_guide, Adam({}), Trace_ELBO())
        with poutine.block():
            for _ in trange(10_000):
                svi.step(state, action)
        cache['transition_logits'][state_n][action_n] = pyro.param(f"transition_preferences_{state_n}_{action_n}")
    return cache['transition_logits'][state_n][action_n]


def util_model(state, action):
    tl = transition_logits(state, action)
    state_dist = dist.Categorical(logits=tl)
    next_state = pyro.sample("state", state_dist)
    pyro.sample("util", dist.Categorical(logits=torch.zeros(len(states))), obs=next_state)
    return util_table[next_state]


def util_guide(state, action):
    tl = transition_logits(state, action)
    state_dist = dist.Categorical(logits=tl)
    next_state = pyro.sample("state", state_dist)
    state_n = index_tensor_to_elem(state, states)
    action_n = index_tensor_to_elem(action, actions)
    preferences = pyro.param(f"util_preferences_{state_n}_{action_n}", torch.zeros(len(states)))
    pyro.sample("util", dist.Categorical(logits=preferences), obs=next_state,
                # infer={'is_auxiliary': True} # TODO: Warning?
                )
    return util_table[next_state]


def util_log_probs(state, action):
    state_n = index_tensor_to_elem(state, states)
    action_n = index_tensor_to_elem(action, actions)
    if 'util_log_probs' not in cache:
        cache['util_log_probs'] = {}
    if state_n not in cache['util_log_probs']:
        cache['util_log_probs'][state_n] = {}
    if action_n not in cache['util_log_probs'][state_n]:
        svi = SVI(util_model, util_guide, Adam(dict()), Trace_ELBO())
        with poutine.block():
            for _ in trange(10_000):
                svi.step(state, action)
        cache['util_log_probs'][state_n][action_n] = pyro.param(f"util_preferences_{state_n}_{action_n}")
    return cache['util_log_probs'][state_n][action_n]


def agent_model(state):
    action = pyro.sample("action", dist.Categorical(logits=torch.zeros(len(actions))))
    u_log_probs_un = util_log_probs(state, action)
    u_probs_un = u_log_probs_un.exp()
    u_probs_sum = u_probs_un.sum()
    u_probs = u_probs_un / u_probs_sum
    expected_util = (u_probs * util_table).mean()
    pyro.factor("weigh_action", expected_util)
    return action


def agent_guide(state):
    preferences = pyro.param("preferences", torch.zeros(len(actions)))
    action = pyro.sample("action", dist.Categorical(logits=preferences))
    return action


def agent_dist(state):
    svi = SVI(agent_model, agent_guide, Adam(dict(lr=.05)), Trace_ELBO())
    for _ in trange(10_000):
        svi.step(state)
    return dist.Categorical(logits=pyro.param("preferences"))


a_dist = agent_dist(tensor(0))

for p in ["preferences", "util_preferences_lost_stay", "util_preferences_lost_switch",
          "transition_preferences_lost_stay", "transition_preferences_lost_switch"]:
    print(f"{p}:", un_log_preferences_to_n_preferences(pyro.param(p)))

for _ in range(3):
    print(index_tensor_to_elem(a_dist.sample(), actions))
