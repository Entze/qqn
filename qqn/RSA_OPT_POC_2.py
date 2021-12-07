import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from torch import Tensor, tensor
from pyro.optim import Adam
from tqdm import trange

"""
"""

torch.manual_seed(0)
pyro.set_rng_seed(0)


def condition(name: str, ten: Tensor):
    f = torch.where(ten, 0.0, float(-100))
    pyro.factor(name, f)


def meaning(utterance, obj):
    return torch.eq(utterance, obj)


def l0_model(utterance):
    obj = pyro.sample("L0_sample", dist.Categorical(logits=torch.zeros(3)))
    m = meaning(utterance, obj)
    condition("L0_cond", m)
    return obj


def l0_guide(utterance):
    preferences = pyro.param("L0_sample_preferences", torch.zeros(3))
    obj = pyro.sample("L0_sample", dist.Categorical(logits=preferences))
    return obj


optim = Adam(dict(lr=0.025))

svi = SVI(l0_model, l0_guide, optim, Trace_ELBO())

for _ in trange(10_000):
    svi.step(tensor(1))

print(pyro.param("L0_sample_preferences"))
