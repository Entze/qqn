from typing import Dict, Any, Iterable, List

import torch
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyroapi import pyro
import pyro
from pyro.optim import Adam
from torch import Tensor, tensor

import pyro.distributions as dist
from tqdm import trange

torch.manual_seed(0)
pyro.set_rng_seed(0)
pyro.clear_param_store()

objects = [
    dict(color="blue", shape="square", string="blue square"),
    dict(color="blue", shape="circle", string="blue circle"),
    dict(color="green", shape="square", string="green square")
]

# IDEA: 'create embedding'?
object_dict_embedding = {
    "positions": ["color", "shape", "string"],
    "color": ["blue", "green"],
    "shape": ["square", "circle"],
    "string": ["blue square", "blue circle", "green square"]
}


def index_of(element, iterable: Iterable) -> int:
    for i, e in enumerate(iterable):
        if e == element:
            return i
    return -1


def object_to_att_tensor(obj: Dict[str, str], dict_embedding: Dict[str, List[str]]) -> Tensor:
    embedding: List[int] = []
    for key in dict_embedding["positions"]:
        if key not in obj:
            embedding.append(-1)
            continue
        assert key in obj
        val = obj[key]
        emb = index_of(val, dict_embedding[key])
        embedding.append(emb)
    return tensor(embedding)


def att_tensor_to_object(ten: Tensor, dict_embedding: Dict[str, List[str]]) -> Dict[str, str]:
    ten = ten.int()
    obj = {}
    for i, e in enumerate(ten):
        pos = dict_embedding["positions"][i]
        obj[pos] = dict_embedding[pos][e]

    return obj


utterances = ["blue", "green", "square", "circle"]


def elem_to_index_tensor(elem, list_embedding: List) -> Tensor:
    return tensor(index_of(elem, list_embedding))


def index_tensor_to_elem(ten: Tensor, list_embedding: List):
    return list_embedding[ten.int().item()]


def condition(name: str, ten, knockout=float('-inf')):
    pyro.factor(name, torch.where(ten, 0.0, -100.0))


def meaning_predicate(utterance, obj):
    k = utterance.int().item()
    return {0: obj[0] == tensor(0),
            1: obj[0] == tensor(1),
            2: obj[1] == tensor(0),
            3: obj[1] == tensor(1)}[k]


# def meaning_predicate_idea(utterance, obj):
#    return torch.where(utterance == 0, obj[0] == 0, torch.where())


def l0_model(utterance, step=None):
    utterance_index = utterance.int().item()
    obj_index = pyro.sample(f"L0_sample_{utterance_index}_{step}", dist.Categorical(logits=torch.zeros(len(objects))))
    obj = object_to_att_tensor(objects[obj_index.int().item()], object_dict_embedding)  # TODO: Vectorize?
    m = meaning_predicate(utterance, obj)
    condition("L0_cond", m)
    return obj_index


def l0_guide(utterance, step=None):
    utterance_index = utterance.int().item()
    preferences = pyro.param(f"L0_sample_{utterance_index}_preferences", torch.zeros(len(objects)))
    obj_index = pyro.sample(f"L0_sample_{utterance_index}_{step}", dist.Categorical(logits=preferences))
    return obj_index


alpha = 1  # TODO


def cost(utterance):
    return 0


cache = {}


def l0(utt):
    if "L0" not in cache:
        cache["L0"] = {}
    utt_index = utt.int().item()  # TODO vectorize
    if utt_index not in cache["L0"]:
        optim = Adam(dict(lr=0.025))
        svi = SVI(l0_model, l0_guide, optim, Trace_ELBO())
        for i in trange(10_000):
            with poutine.block():
                svi.step(utt, i)
            cache["L0"][utt_index] = pyro.param(f"L0_sample_{utt_index}_preferences")
    return cache["L0"][utt_index]


def s0_model(obj, step=None):
    # obj is just a index to objects[obj]
    obj_index = obj.int().item()
    utt = pyro.sample(f"S0_sample_{obj_index}_{step}", dist.Categorical(logits=torch.zeros(len(utterances))))
    l0_dist = dist.Categorical(logits=l0(utt))
    score = l0_dist.log_prob(obj)
    pyro.factor("S0_factor", alpha * score - cost(utt))
    return utt


def s0_guide(obj, step=None):
    # obj is just a index to objects[obj]
    obj_index = obj.int().item()
    preferences = pyro.param(f"S0_sample_{obj_index}_preferences", torch.zeros(len(utterances)))
    utt = pyro.sample(f"S0_sample_{obj_index}_{step}", dist.Categorical(logits=preferences))
    return utt


def s0(obj):
    if "S0" not in cache:
        cache["S0"] = {}
    obj_index = obj.int().item()  # TODO vectorize
    if obj_index not in cache["S0"]:
        optim = Adam(dict(lr=0.025))
        svi = SVI(s0_model, s0_guide, optim, Trace_ELBO())
        for i in trange(10_000):
            with poutine.block():
                svi.step(obj, i)
            cache["S0"][obj_index] = pyro.param(f"S0_sample_{obj_index}_preferences")
    return cache["S0"][obj_index]


def l1_model(utterance, step=None):
    # obj is just a index to objects[obj]
    utterance_index = utterance.int().item()
    obj = pyro.sample(f"L1_sample_{utterance_index}_{step}", dist.Categorical(logits=torch.zeros(len(objects))))
    s0_dist = dist.Categorical(logits=s0(obj))
    score = s0_dist.log_prob(utterance)
    pyro.factor("L1_factor", alpha * score - cost(obj))
    return obj


def l1_guide(utterance, step=None):
    # obj is just a index to objects[obj]
    utterance_index = utterance.int().item()
    preferences = pyro.param(f"L1_sample_{utterance_index}_preferences", torch.zeros(len(objects)))
    obj = pyro.sample(f"L1_sample_{utterance_index}_{step}", dist.Categorical(logits=preferences))
    return obj


def l1(utterance):
    if "L1" not in cache:
        cache["L1"] = {}
    utterance_index = utterance.int().item()  # TODO vectorize
    if utterance_index not in cache["L1"]:
        optim = Adam(dict(lr=0.025))
        svi = SVI(l1_model, l1_guide, optim, Trace_ELBO())
        for i in trange(10_000):
            with poutine.block():
                svi.step(utterance, i)
            cache["L1"][utterance_index] = pyro.param(f"L1_sample_{utterance_index}_preferences")
    return cache["L1"][utterance_index]


for utterance in utterances:
    print()
    print(utterance, l0(elem_to_index_tensor(utterance, utterances)))
    print()

for obj in objects:
    print()
    print(obj, s0(elem_to_index_tensor(obj, objects)))
    print()

for utterance in utterances:
    print()
    print(utterance, l1(elem_to_index_tensor(utterance, utterances)))
    print()
