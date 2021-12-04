# possible states of the world
import pyro
import pyro.infer
import torch
import pyro.distributions as dist
from torch import tensor

from qqn.webppl import viz

rsa_cache = dict(literal_listener={}, pragmatic_speaker={}, pragmatic_listener={})

states = [0, 1, 2, 3]

state_dist = dist.Categorical(logits=torch.zeros(len(states)))


def state_prior(name):
    return pyro.sample(name, state_dist)


# sample a state
print(state_dist.sample())

utterances = ['all', 'some', 'none']

utt_dist = dist.Categorical(logits=torch.zeros(len(utterances)))


# possible utterances
def utterance_prior(name):
    return pyro.sample(name, utt_dist)


# meaning function to interpret the utterances
literal_meanings = dict(
    all=lambda state: state == 3,
    some=lambda state: state > 0,
    none=lambda state: state == 0)

# sample an utterance
utt_t = utt_dist.sample()
utt = list(literal_meanings.keys())[int(utt_t.item())]

# get its meaning
meaning = literal_meanings[utt]

# apply meaning to state = 3
print([utt, meaning(3)])


def condition(name: str, val: bool = False):
    pyro.factor(name, tensor(float(0 if val else '-inf')))


# literal listener
def literal_listener(utt, num_samples=10):
    if utt not in rsa_cache["literal_listener"]:
        def model():
            state_t = state_prior("lit_state")
            state = int(state_t.item())
            meaning = literal_meanings[utt]
            condition("lit_cond", meaning(state))
            return state_t

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["literal_listener"][utt] = marginal

    return rsa_cache["literal_listener"][utt]


print("literal listener's interpretation of 'some':")
viz(literal_listener("some", num_samples=1000))

cost = dict(
    all=1,
    some=1,
    none=1)


# set speaker optimality

# pragmatic speaker
def pragmatic_speaker(state, alpha=1., num_samples=10):
    if state not in rsa_cache["pragmatic_speaker"]:
        def model():
            utt_t = utterance_prior("prag_utt")
            utt = list(literal_meanings.keys())[int(utt_t.item())]
            alpha_t = tensor(alpha).float()
            state_t = tensor(state).float()
            lit_lis = literal_listener(utt, num_samples=num_samples)
            state_prob = lit_lis.log_prob(state_t)
            pyro.factor("speak_factor", alpha_t * (state_prob - tensor(cost.get(utt, 0))))
            return utt_t

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["pragmatic_speaker"][state] = marginal
    return rsa_cache["pragmatic_speaker"][state]


print("speaker's production probabilities for state 3:")
viz(pragmatic_speaker(3, num_samples=100))


def utt_to_tensor(utt, utts):
    for k, v in enumerate(utts):
        if v == utt:
            return tensor(k).float()


# pragmatic listener
def pragmatic_listener(utt, num_samples=10):
    if utt not in rsa_cache["pragmatic_listener"]:
        def model():
            state_t = state_prior("prag_state")
            state = int(state_t.item())
            utt_t = utt_to_tensor(utt, utterances)
            pyro.sample("speak_utt", pragmatic_speaker(state, num_samples=num_samples), obs=utt_t)
            return state_t

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["pragmatic_listener"][utt] = marginal
    return rsa_cache["pragmatic_listener"][utt]


print("pragmatic listener's interpretation of 'some':")
viz(pragmatic_listener('some', num_samples=10_000))


