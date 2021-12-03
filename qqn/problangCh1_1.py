from collections import OrderedDict

import torch
from pyroapi import pyro
import pyro.infer
import pyro.distributions as dist
from torch import tensor

from qqn.webppl import viz

colors = OrderedDict(blue=0, green=1)
shapes = OrderedDict(square=0, circle=1)
strings = OrderedDict({'blue square': 0, 'blue circle': 1, 'green square': 2})

objects = [dict(color="blue", shape="square", string="blue square"),
           dict(color="blue", shape="circle", string="blue circle"),
           dict(color="green", shape="square", string="green square")]

utterances = ["blue", "green", "square", "circle"]

obj_prior_dist = dist.Categorical(tensor([0.2, 0.7, 0.1]))


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


def meaning(utterance, obj):
    return utterance in obj.values()


def condition(name: str, val: bool = False):
    pyro.factor(name, tensor(float(0 if val else '-inf')))


rsa_cache = dict(literal_listener={})


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


dist = literal_listener("blue")
for i in range(10):
    print(tensor_to_obj(dist.sample()))
