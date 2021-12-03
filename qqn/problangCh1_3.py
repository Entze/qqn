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
    for i in range(len(objects)):
        if objects[i] == obj:
            return torch.as_tensor(i).float()
    raise


def tensor_to_obj(ten):
    obj = objects[int(ten.item())]
    return obj


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
            obj_t = pyro.sample("Basic event", obj_prior_dist)
            obj = tensor_to_obj(obj_t)
            utt_truth_val = meaning(utterance, obj)
            condition("Basic observation", utt_truth_val)
            return obj_t

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
            lit_lis_sample = lit_lis.sample()
            literal_listener_score_t = lit_lis.log_prob(obj_t).exp()
            cost_of_utterance = cost(utterance)
            cost_of_utterance_t = torch.as_tensor(cost_of_utterance).float()
            # vvv SOFTMAX vvv
            #pyro.factor("First Order Observation", alpha_t * (literal_listener_score_t - cost_of_utterance_t))
            condition("First Order Observation", lit_lis_sample == obj_t)
            # ^^^ ARGMAX ^^^
            return utterance_t

        importance = pyro.infer.Importance(model=model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache['pragmatic_speaker'][obj] = marginal
    return rsa_cache['pragmatic_speaker'][obj]


dist = literal_listener("red")
for i in range(10):
    print(tensor_to_obj(dist.sample()))

dist = pragmatic_speaker(objects[3])

viz(dist)

for i in range(10):
    print(tensor_to_utterance(dist.sample()))
