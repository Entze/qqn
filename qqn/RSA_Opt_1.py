from typing import Dict, Any, Iterable, List

import torch
from pyro.infer import SVI, Trace_ELBO
from pyroapi import pyro
import pyro
from pyro.optim import Adam
from torch import Tensor, tensor

import pyro.distributions as dist
from tqdm import trange

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


def object_to_tensor(obj: Dict[str, str], dict_embedding: Dict[str, List[str]]) -> Tensor:
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


def tensor_to_object(ten: Tensor, dict_embedding: Dict[str, List[str]]) -> Dict[str, str]:
    ten = ten.int()
    obj = {}
    for i, e in enumerate(ten):
        pos = dict_embedding["positions"][i]
        obj[pos] = dict_embedding[pos][e]

    return obj


utterances = ["blue", "green", "square", "circle"]


def elem_to_tensor(elem: str, list_embedding: List[str]) -> Tensor:
    return tensor(index_of(elem, list_embedding))


def tensor_to_elem(ten: Tensor, list_embedding: List[str]) -> str:
    return list_embedding[ten.int().item()]


def condition(name: str, ten):
    pyro.factor(name, torch.where(ten, 0.0, -100.0))


def meaning(utterance: Tensor, obj: Tensor) -> Tensor:
    # Tensor([5, 2, 3]) -> Tensor([1, 5, 3]) -> Tensor([True, False, True)]
    # https://stackoverflow.com/questions/69225949/pytorch-tensor-how-to-get-the-index-of-a-tensor-given-a-multidimensional-tenso
    sorted_obj, indices = torch.sort(obj)
    index_into_sorted_obj = torch.searchsorted(sorted_obj, utterance)
    return index_into_sorted_obj < indices.size(dim=0)


def l0_model(utterance):
    obj_index = pyro.sample("L0_sample", dist.Categorical(logits=torch.zeros(len(objects))))
    obj = pyro.deterministic("L0_sample_mapped", object_to_tensor(objects[obj_index.int().item()],
                                                                  object_dict_embedding))  # TODO: Vectorize?
    m = meaning(utterance, obj)
    condition("L0_cond", m)
    return obj


def l0_guide(utterance):
    preferences = pyro.param("L0_sample_preferences", torch.zeros(len(objects)))
    obj_index = pyro.sample("L0_sample", dist.Categorical(logits=preferences))
    obj = pyro.deterministic("L0_sample_mapped", object_to_tensor(objects[obj_index.int().item()],
                                                                  object_dict_embedding))  # TODO: Vectorize?
    return obj


optim = Adam(dict(lr=0.025))

svi = SVI(l0_model, l0_guide, optim, Trace_ELBO())

for _ in trange(10_000):
    svi.step(elem_to_tensor("blue", utterances))

print(pyro.param("L0_sample_preferences"))
