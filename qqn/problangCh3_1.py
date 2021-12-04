import math
from itertools import repeat

import pyro.distributions as dist

# Prior probability of kettle prices (taken from human experiments)
import torch
from frozendict import frozendict
from pyroapi import pyro
from torch import tensor

from qqn.webppl import viz

rsa_cache = dict(literal_listener={}, pragmatic_speaker={}, pragmatic_listener={})


def approx(x):
    return round(x, 1)


fullState = dict(price=51, valence=True)
print("full state =", fullState)

qud_fns = dict(
    price=lambda state: frozendict(price=state["price"]),
    valence=lambda state: frozendict(valence=state["valence"]),
    priceValence=lambda state: frozendict(price=state["price"], valence=state["valence"]),
    approxPrice=lambda state: frozendict(price=approx(state["price"])),
    approxPriceValence=lambda state: frozendict(price=approx(state["price"]), valence=approx(state["valence"])),
)

valenceQudFn = qud_fns["valence"]
priceQudFn = qud_fns["price"]
priceValenceQudFn = qud_fns["priceValence"]
approxPriceQudFn = qud_fns["approxPrice"]

prices = [
    50, 51,
    500, 501,
    1000, 1001,
    5000, 5001,
    10000, 10001]

prices_probs = [
    0.4205, 0.3865,
    0.0533, 0.0538,
    0.0223, 0.0211,
    0.0112, 0.0111,
    0.0083, 0.0120]

price_dist = dist.Categorical(tensor(prices_probs))


def price_prior():
    sample_t = price_dist.sample()
    return prices[sample_t.item()]


valence_probs = {
    50: 0.3173,
    51: 0.3173,
    500: 0.7920,
    501: 0.7920,
    1000: 0.8933,
    1001: 0.8933,
    5000: 0.9524,
    5001: 0.9524,
    10000: 0.9864,
    10001: 0.9864
}


def valence_prior(price):
    sample_t = dist.Bernoulli(valence_probs[price]).sample()
    return bool(sample_t.item())


# print([price_dist.log_prob(tensor(price)).exp() for price in valence_probs])


# Prior over QUDs
qud_dist = dist.Categorical(logits=torch.zeros(len(qud_fns)))


def qud_prior():
    sample_t = qud_dist.sample()
    return list(qud_fns.keys())[int(sample_t.item())]


utterances = [
    50, 51,
    500, 501,
    1000, 1001,
    5000, 5001,
    10000, 10001
]

utterances_dist = dist.Categorical(logits=torch.zeros(len(utterances)))


def utterances_prior():
    sample_t = utterances_dist.sample()
    return tensor_to_utterance(sample_t)


def cost(utterance):
    if utterance == approx(utterance):
        return 0
    return 1


# Literal interpretation "meaning" function;
# checks if uttered number reflects price state
def meaning(utterance, price):
    return utterance == price


def condition(name: str, val: bool = False):
    pyro.factor(name, tensor(float(0 if val else '-inf')))


def state_to_tensor(state):
    price = state.get("price", -1.)
    valence = state.get("valence", -1.)
    return tensor([price, valence])


def tensor_to_state(tens):
    state_l = tens.tolist()
    state = dict()
    if state_l[0] >= 0:
        state["price"] = state_l[0]
    if state_l[1] >= 0:
        state["valence"] = state_l[1]
    return state


def utterance_to_tensor(utterance):
    for i, u in enumerate(utterances):
        if u == utterance:
            return tensor(i).float()
    return tensor(-1.)


def tensor_to_utterance(tensor):
    return utterances[int(tensor.item())]


# literal listener
def literal_listener(utt, qud, num_samples=10):
    if utt not in rsa_cache["literal_listener"]:
        rsa_cache["literal_listener"][utt] = {}
    if qud not in rsa_cache["literal_listener"][utt]:
        def model():
            price = price_prior()
            valence = valence_prior(price)
            full_state = frozendict(price=price, valence=valence)
            qud_fn = qud_fns[qud]
            qud_answer = qud_fn(full_state)
            qud_meaning = meaning(utt, price)
            condition("lit_cond", qud_meaning)
            return state_to_tensor(qud_answer)

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["literal_listener"][utt][qud] = marginal

    return rsa_cache["literal_listener"][utt][qud]


# Speaker, chooses an utterance to convey a particular answer of the qud

def pragmatic_speaker(full_state, qud, alpha=1., num_samples=10):
    if full_state not in rsa_cache["pragmatic_speaker"]:
        rsa_cache["pragmatic_speaker"][full_state] = {}
    if qud not in rsa_cache["pragmatic_speaker"][full_state]:
        def model():
            utterance = utterances_prior()
            qud_fn = qud_fns[qud]
            qud_answer = qud_fn(fullState)
            alpha_t = tensor(alpha).float()
            lit_lis = literal_listener(utterance, qud, num_samples=num_samples)
            state_prob = lit_lis.log_prob(state_to_tensor(qud_answer))

            pyro.factor("speak_factor", alpha_t * (state_prob - tensor(cost(utterance))))
            return utterance_to_tensor(utterance)

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["pragmatic_speaker"][full_state][qud] = marginal
    return rsa_cache["pragmatic_speaker"][full_state][qud]


# Pragmatic listener, jointly infers the price state, speaker valence, and QUD

# pragmatic listener
def pragmatic_listener(utterance, num_samples=10):
    if utterance not in rsa_cache["pragmatic_listener"]:
        def model():
            # //////// priors ////////
            price = price_prior()
            valence = valence_prior(price)
            qud = qud_prior()
            # ////////////////////////
            full_state = frozendict(price=price, valence=valence)
            prag_speak = pragmatic_speaker(full_state, qud, num_samples=num_samples)
            pyro.sample("speak_utt", prag_speak, obs=utterance_to_tensor(utterance))
            return state_to_tensor(full_state)

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["pragmatic_listener"][utterance] = marginal
    return rsa_cache["pragmatic_listener"][utterance]


print("pragmatic listener's joint interpretation of 'The kettle cost $10,000':")
listener_posterior = pragmatic_listener(10000, num_samples=1_000)

print(listener_posterior.sample())

# viz(listener_posterior)
