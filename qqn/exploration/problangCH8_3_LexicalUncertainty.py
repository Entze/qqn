import math

from pyroapi import pyro
import pyro.distributions as dist
import torch
from torch import tensor

# fold:
from itertools import combinations, chain

from qqn.exploration.webppl import viz


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


worlds = ["A", "B", "AB"]

print(list(powerset(worlds)))

belief_states = filter(
    lambda x: len(x) > 0,
    powerset(worlds)
)

print(list(belief_states))

speaker_knowledgeability_states = [0, 1, 2, 3]


def knowledgeability_level_prior():
    return dist.Categorical(logits=tensor(speaker_knowledgeability_states)).sample()


def belief_state_prior(speaker_knowledgeability_level):
    weights = [math.exp(- speaker_knowledgeability_level * len(bs)) for bs in list(belief_states)]
    return dist.Categorical(logits=tensor(weights)).sample()
    # eigentlich return categorical({vs: belief_states, ps: weights})


utterances = [
    'Anne',
    'Bob',
    'both',
    'Anne or Bob',
    'Anne or both',
    'Bob or both',
    'Anne or Bob or both'
]

cost_disjunction = 0.2
cost_conjunction = 0.1


def tensor_to_utterance(utterance_t):
    return utterances[int(utterance_t.item())]


def utterance_cost(utterance_t):
    utterance = tensor_to_utterance(utterance_t)
    utt_cost_table = {
        "Anne": 0,
        'Bob': 0,
        'both': cost_conjunction,
        'Anne or Bob': cost_disjunction,
        'Anne or both': cost_disjunction + cost_conjunction,
        'Bob or both': cost_disjunction + cost_conjunction,
        'Anne or Bob or both': 2 * cost_disjunction + cost_conjunction
    }
    return utt_cost_table[utterance]


alpha = 5


def utterance_prior(num_samples=10):
    def model():
        utterance = dist.Categorical(logits=torch.zeros(len(utterances))).sample()
        alpha_t = tensor(alpha).float()
        pyro.factor("utt_factor", - alpha_t * utterance_cost(utterance))
        return utterance

    importance = pyro.infer.Importance(model, num_samples=num_samples)
    importance.run()
    marginal = pyro.infer.EmpiricalMarginal(importance)
    return marginal


lexica = [{"Anne": "only Anne", "Bob": "only Bob"},
          {"Anne": "Anne or more", "Bob": "Bob or more"}]


def lexicon_prior():
    return dist.Categorical(logits=tensor(len(lexica))).sample()


def utterance_meaning(utterance_t, lexicon):  # why? belief_state?
    utterance = tensor_to_utterance(utterance_t)
    basic_meaning = {
        "Anne": ["A"] if lexicon["Anne"] == "only Anne" else ["A", "AB"],
        "Bob": ["B"] if lexicon["Bob"] == "only Bob" else ["B", "AB"],
        "both": ["AB"],
        "Anne or Bob": ["A", "B"] if lexicon["Anne"] == "only Anne" and lexicon["Bob"] == "only Bob" else worlds,
        "Anne or both": ["A", "AB"],
        "Bob or both": ["B", "AB"],
        'Anne or Bob or both': worlds
    }
    return basic_meaning[utterance]


def literal_listener(utterance, lexicon, num_samples=10):
    def model():
        world = dist.Categorical(logits=tensor(utterance_meaning(utterance, lexicon))).sample()
        return world

    importance = pyro.infer.Importance(model, num_samples=num_samples)
    importance.run()
    marginal = pyro.infer.EmpiricalMarginal(importance)
    return marginal


def utility(belief_state, utterance, lexicon):
    scores = [literal_listener(utterance, lexicon).log_prob(bs) for bs in belief_state]
    return 1 / len(belief_state) * sum(scores)


def speaker(belief_state, lexicon, alpha=1., num_samples=10):
    def model():
        utterance = utterance_prior().sample()
        lit_lis = literal_listener(utterance, lexicon)
        # for what do we need the lit_lis

        util = utility(belief_state, utterance, lexicon)
        alpha_t = tensor(alpha).float()

        # this factor might miss the lit_lis:
        pyro.factor("speak_factor", alpha_t * util)
        return utterance

    importance = pyro.infer.Importance(model, num_samples=num_samples)
    importance.run()
    marginal = pyro.infer.EmpiricalMarginal(importance)
    return marginal


# viz(literal_speaker(["A"]))


def listener(utterance):
    def model():
        knowledgeability = knowledgeability_level_prior()
        # felher auf der website ?: knowledgeability_level_prior(speaker_knowledgeability_states)
        lexicon = lexicon_prior()
        belief_state = belief_state_prior(knowledgeability)
        speak = speaker(belief_state, lexicon)
        pyro.factor("speak_factor", lit_speak.log_prob(utterance))
        return torch.stack(belief_state, lexicon)

    importance = pyro.infer.Importance(model, num_samples=num_samples)
    importance.run()
    marginal = pyro.infer.EmpiricalMarginal(importance)
    return marginal


viz(listener("Anne"))
viz(listener("Anne or both"))
