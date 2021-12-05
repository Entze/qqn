import math

import pyroapi
from frozenlist import FrozenList
from pyroapi import pyro
import pyro.distributions as dist
import torch
from torch import tensor, Tensor, erf

# possible object weights

objects = [2, 3, 4];


def object_prior():
    return dist.Categorical(logits=torch.zeros(len(objects))).sample()


number_objects = 3


# build states with n many objects
def state_prior(n_obj_left, state_so_far):
    state_so_far = [] if state_so_far is None else state_so_far
    if n_obj_left == 0:
        return state_so_far
    else:
        new_obj = object_prior()
        new_state = state_so_far.concat([new_obj])
        return state_prior(n_obj_left - 1, new_state)


# threshold priors
def dist_theta_prior():
    sample_t = dist.Categorical(logits=torch.zeros(len(objects))).sample()
    return sample_t - 1  # 1 minus possible object values


def coll_theta_prior():
    return state_prior(number_objects).sum() - 1  # 1 minus possible state sums


utterances = [
    "null",
    "heavy",
    "each-heavy",
    "together-heavy"
]


# costs: null < ambiguous < unambiguous
def utterance_prior():
    return dist.Categorical(logits=torch.zeros(len(utterances))).sample()


def tensor_to_utterance(utterance_t):
    return utterances[int(utterance_t.item())]


def cost(utterance_t):
    utterance = tensor_to_utterance(utterance_t)
    if utterance == "null":
        return 0
    elif utterance == "heavy":
        return 1
    return 2


# x > theta interpretations
'''
def coll_interpretation(state, coll_theta):
    return state.sum() > coll_theta
'''


def coll_interpretation(state, coll_theta, noise):
    weight = 1 - (0.5 * (1 + erf((coll_theta - state.sum()) /
                                 (noise * math.sqrt(2)))))
    return dist.Bernoulli(weight).sample()


def dist_interpretation(state: Tensor, dist_theta):
    return torch.all(state > dist_theta)


# meaning function
def meaning(utt_t, state, dist_theta, coll_theta, is_collective):
    utt = tensor_to_utterance(utt_t)
    if utt == "null":
        return True
    elif utt == "each-heavy":
        return dist_interpretation(state, dist_theta)
    elif utt == "together-heavy":
        return coll_interpretation(state, coll_theta)
    elif is_collective:
        return coll_interpretation(state, coll_theta)
    return dist_interpretation(state, dist_theta)

