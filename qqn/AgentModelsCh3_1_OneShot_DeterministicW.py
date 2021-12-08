from typing import Iterable, List

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.infer import Trace_ELBO, SVI
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


########################################################################################################


actions = ['italian', 'french']
states = ["pizza", "steak frites"]


def action_to_tensor(act):
    return elem_to_index_tensor(act, actions)


def state_to_tensor(state):
    return elem_to_index_tensor(state, states)


def state_t_to_name(state_t):
    return index_tensor_to_elem(state_t, states)


def transition(state, action):
    if action == action_to_tensor('italian'):
        return state_to_tensor('pizza')
    return state_to_tensor('steak frites')


def agent_model(state):
    action = pyro.sample("action", dist.Categorical(logits=torch.zeros(len(actions))))
    next_state = transition(state, action)
    m = next_state == state_to_tensor('pizza')
    condition("condition", m)
    return action


def agent_guide(state):
    preferences = pyro.param("preferences", torch.zeros(len(actions)))
    action = pyro.sample("action", dist.Categorical(logits=preferences))
    return action


def agent_dist(state):
    svi = SVI(agent_model, agent_guide, Adam(dict(lr=.05)), Trace_ELBO())
    for _ in trange(1000):
        svi.step(state)
    return dist.Categorical(logits=pyro.param("preferences"))


a_dist = agent_dist(tensor(0))

print(pyro.param("preferences"))

for _ in range(3):
    print(index_tensor_to_elem(a_dist.sample(), actions))
