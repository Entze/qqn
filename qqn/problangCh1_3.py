from collections import OrderedDict

import torch
from frozendict import frozendict
from pyroapi import pyro
import pyro.infer
import pyro.distributions as dist
from torch import tensor

from qqn.webppl import viz

colors = OrderedDict(blue=0, green=1, red=2)
shapes = OrderedDict(square=0, circle=1, triangle=2)
strings = OrderedDict({'blue square': 0, 'blue circle': 1, 'green square': 2, 'red triangle': 3})

objects = [frozendict(dict(color="blue", shape="square", string="blue square")),
           frozendict(dict(color="blue", shape="circle", string="blue circle")),
           frozendict(dict(color="green", shape="square", string="green square")),
           frozendict(dict(color="red", shape="triangle", string="red triangle"))]

utterances = ["blue", "green", "square", "circle", "red", "triangle"]

obj_prior_dist = dist.Categorical(logits=torch.zeros(len(objects)))

utterance_prior_dist = dist.Categorical(logits=torch.zeros(len(utterances)))

torch.manual_seed(0)
pyro.set_rng_seed(0)


def object_prior():
    obj = obj_prior_dist.sample()
    return objects[obj.item()]


def obj_to_tensor(obj):
    color_dec = float(colors[obj['color']])
    shape_dec = float(shapes[obj['shape']])
    string_dec = float(strings[obj['string']])
    return tensor([color_dec, shape_dec, string_dec])


def tensor_to_obj(ten):
    color_int = int(ten[0].item())
    shape_int = int(ten[1].item())
    string_int = int(ten[2].item())
    return dict(color=list(colors.keys())[color_int], shape=list(shapes.keys())[shape_int],
                string=list(strings.keys())[string_int])


def tensor_to_utterance(ten):
    return utterances[int(ten.item())]


def meaning(utterance, obj):
    return utterance in obj.values()


def cost(utterance):
    return 0.0


def condition(name: str, val: bool = False):
    pyro.factor(name, tensor(float(0 if val else '-inf')))


rsa_cache = dict(literal_listener={}, pragmatic_speaker={})


def literal_listener(utterance, num_samples=100):
    if utterance not in rsa_cache['literal_listener']:
        def model():
            obj = object_prior()
            utt_truth_val = meaning(utterance, obj)
            condition("Basic evidence", utt_truth_val)
            return obj_to_tensor(obj)

        importance = pyro.infer.Importance(model=model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache['literal_listener'][utterance] = marginal
    return rsa_cache['literal_listener'][utterance]


def pragmatic_speaker(obj, num_samples=100, alpha=1.0):
    if obj not in rsa_cache['pragmatic_speaker']:
        alpha_t = torch.as_tensor(alpha).float()

        def model():
            utterance_t = pyro.sample("First Order Event", utterance_prior_dist)
            utterance = tensor_to_utterance(utterance_t)
            obj_t = obj_to_tensor(obj)
            lit_lis = literal_listener(utterance)
            literal_listener_score_t = lit_lis.log_prob(obj_t).exp()
            cost_of_utterance = cost(utterance)
            cost_of_utterance_t = torch.as_tensor(cost_of_utterance).float()
            pyro.factor("First Order Observation", alpha_t * (literal_listener_score_t - cost_of_utterance_t))
            return utterance_t

        importance = pyro.infer.Importance(model=model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache['pragmatic_speaker'][obj] = marginal
    return rsa_cache['pragmatic_speaker'][obj]


dist = literal_listener("blue")
for i in range(10):
    print(tensor_to_obj(dist.sample()))

dist = pragmatic_speaker(objects[3])

viz(dist)

for i in range(10):
    print(tensor_to_utterance(dist.sample()))
