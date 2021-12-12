from collections import OrderedDict

import torch
from pyroapi import pyro
import pyro.infer
import pyro.distributions as dist

from qqn.exploration.webppl import viz

actions = OrderedDict(a1=0, a2=1, a3=0)

utility_dict = dict(
    a1=-1,
    a2=6,
    a3=8
)

action_dist = dist.Categorical(logits=torch.zeros(len(actions)))


def from_dict_to_tensor(obj, d):
    n = d[obj]
    return torch.as_tensor(n).float()


def from_tensor_with_dict(ten, d):
    k = int(ten.item())
    return list(d.keys())[k]


def action_to_tensor(action):
    return from_dict_to_tensor(action, actions)


def tensor_to_action(action):
    return from_tensor_with_dict(action, actions)


def utility(action):
    return float(utility_dict.get(action, '-inf'))


def agent(alpha=1.0, num_samples=100):
    alpha_t = torch.as_tensor(alpha)

    def model():
        action_t = pyro.sample("Action", action_dist)
        action = tensor_to_action(action_t)
        util = utility(action)
        util_t = torch.as_tensor(util)
        pyro.factor("Factor", alpha_t * util_t)
        return action_t

    importance = pyro.infer.Importance(model, num_samples=num_samples)
    importance.run()
    marginal = pyro.infer.EmpiricalMarginal(importance)
    return marginal


viz(agent(alpha=1, num_samples=10_000))
