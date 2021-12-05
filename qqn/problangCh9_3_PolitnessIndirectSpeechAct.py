import math
from frozenlist import FrozenList
from pyroapi import pyro
import pyro
import pyro.infer
import pyro.distributions as dist
import torch
from torch import tensor
from itertools import combinations, chain
from qqn.webppl import viz, expectation

states = [0, 1, 2, 3]

torch.manual_seed(0)
pyro.set_rng_seed(0)

# value function scales social utility by a parameter lambda
lam = 1.25  # value taken from MAP estimate from Yoon, Tessler, et al. 2016


def value_function(s):
    return lam * s


def condition(name: str, val: bool = False):
    pyro.factor(name, tensor(float(0 if val else '-inf')))


literalSemantics = {
    "not_amazing": [0.9652, 0.9857, 0.7873, 0.0018],
    "not_bad": [0.0967, 0.365, 0.7597, 0.9174],
    "not_good": [0.9909, 0.736, 0.2552, 0.2228],
    "not_terrible": [0.2749, 0.5285, 0.728, 0.9203],
    "yes_amazing": [4e-04, 2e-04, 0.1048, 0.9788],
    "yes_bad": [0.9999, 0.8777, 0.1759, 0.005],
    "yes_good": [0.0145, 0.1126, 0.9893, 0.9999],
    "yes_terrible": [0.9999, 0.3142, 0.0708, 0.0198]
}


def flip(p=0.5):
    return dist.Bernoulli(p).sample()


def meaning(words_t, state_t):
    words = list(literalSemantics.keys())[int(words_t.item())]
    state = int(state_t.item())
    return flip(literalSemantics[words][state])


# note in this case that visualizing the joint distribution via viz()
# produces the wrong joint distribution. this is a bug in the viz() program.
# we visualize the marginal distributions instead:

state_dist = dist.Categorical(logits=torch.zeros(len(states)))


def listener0(utterance, num_samples=10):
    def model():
        state = state_dist.sample()
        m = meaning(utterance, state)
        condition("list0_cond", m)
        return state.float()

    importance = pyro.infer.Importance(model, num_samples=num_samples)
    importance.run()
    marginal = pyro.infer.EmpiricalMarginal(importance)
    return marginal


def speaker1(state, phi, speakerOptimality=4.0, num_samples=10):
    def model():
        utterance_t = dist.Categorical(logits=torch.zeros(len(literalSemantics))).sample()
        l0 = listener0(utterance_t, num_samples=num_samples)
        utilities = dict(
            inf=l0.log_prob(state),  # log (s | u)
            soc=expectation(l0),  # E[s]
        )
        speakerUtility = phi * utilities["inf"] + \
                         (1 - phi) * utilities["soc"] - cost(utterance_t)

        pyro.factor("speaker1_factor", speakerOptimality * speakerUtility)
        return utterance_t

    importance = pyro.infer.Importance(model, num_samples=num_samples)
    importance.run()
    marginal = pyro.infer.EmpiricalMarginal(importance)
    return marginal


def listener1(utterance, num_samples=10):
    def model():
        phi = pyro.sample("l1_goal", dist.Uniform(0, 1))
        state = pyro.sample("l1_state", dist.Categorical(logits=torch.zeros(len(states)))).float()
        s1 = speaker1(state, phi, num_samples=num_samples)

        pyro.sample("list1_obs", s1, obs=utterance)

        return torch.stack([state, phi])

    importance = pyro.infer.Importance(model, num_samples=num_samples)
    importance.run()
    marginal = pyro.infer.EmpiricalMarginal(importance)
    return marginal


def utterance_to_tensor(utt):
    for i, u in enumerate(literalSemantics.keys()):
        if u == utt:
            return tensor(i).float()


def cost(utterance_t):
    return tensor(1.)


def speaker2(state, phi, weights, speaker_optimality=4.0, num_samples=10):
    def model():
        utterance_t = dist.Categorical(logits=torch.zeros(len(literalSemantics))).sample()
        l1 = listener1(utterance_t, num_samples=num_samples)
        l1_state = l1.marginal("l1_state")
        l1_goals = l1.marginal("l1_goals")

        util = dict(
            inf=l1_state.log_prob(state),
            soc=expectation(l1_state),
            pres=l1_goals.log_prob(phi)
        )

        total_util = weights["soc"] * util["soc"] + weights["pres"] * util["pres"] + weights["inf"] * util[
            "inf"] - cost(utterance_t)

        speaker_optimality_t = tensor(speaker_optimality)

        pyro.factor("speak2_factor", speaker_optimality_t * total_util)

        utterance = list(literalSemantics.keys())[int(utterance_t.item())]
        utt = utterance.split("_")
        return tensor([utterance_to_tensor(u) for u in utt])

    importance = pyro.infer.Importance(model, num_samples=num_samples)
    importance.run()
    marginal = pyro.infer.EmpiricalMarginal(importance)
    return marginal


s2 = speaker2(0, 0.5, {"soc": 0.05, "pres": 0.60, "inf": 0.35}, num_samples=20)
