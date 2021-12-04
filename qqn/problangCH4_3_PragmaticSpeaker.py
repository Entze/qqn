from pyroapi import pyro
import pyro.distributions as dist
import torch
from torch import tensor

# cache
rsa_cache = dict(literal_listener={}, literal_speaker={}, pragmatic_listener={}, pragmatic_speaker={})


# condition
def condition(name: str, val: bool = False):
    pyro.factor(name, tensor(float(0 if val else '-inf')))


# possible utterances: saying nothing or asserting the ambiguous utterance
utterances = ["null", "every-not"]
utterances_dist = dist.Categorical(logits=torch.zeros(len(utterances)))


def tensor_to_utterances(tens):
    return utterances[int(tens.item())]


def utterance_to_tensor(utterance):
    for i, u in enumerate(utterance):
        if u == utterance:
            return tensor(i).float()


def utterances_prior():
    sample_t = utterances_dist.sample()
    return tensor_to_utterances(sample_t)


# costs
def cost(utterance):
    return 1


# possible world states: how many apples are red
states = [0, 1, 2, 3]
states_dist = dist.Categorical(logits=torch.zeros(len(states)))


def tensor_to_state(tens):
    return states[int(tens.item())]


def state_to_tensor(state):
    for i, s in enumerate(states):
        if s == state:
            return tensor(i).float()


def state_prior():
    sample_t = states_dist.sample()
    return tensor_to_state(sample_t)


# possible scopes
scopes = ["surface", "inverse"]
scopes_dist = dist.Categorical(logits=torch.zeros(len(scopes)))


def tensor_to_scope(tens):
    return scopes[int(tens.item())]


def scope_prior():
    sample_t = scopes_dist.sample()
    return tensor_to_scope(sample_t)


# meaning function
def meaning(utterance, state, scope):
    if utterance == "every-not":
        if scope == "surface":
            return state == 0
        else:
            return state < 3
    else:
        return True


print(meaning("every-not", 1, "surface"))

# QUDs
quds = ["how many?", "all red?"]
quds_dist = dist.Categorical(logits=torch.zeros(len(quds)))


def tensor_to_quds(tens):
    return quds[int(tens.item())]


def quds_prior():
    sample_t = quds_dist.sample()
    return tensor_to_quds(sample_t)


def qud_fun(qud, state):
    if qud == "all red?":
        return state == 3
    else:
        return state


# Literal listener (L0)

def literal_listener(utterance, scope, qud, num_samples=10):
    if utterance not in rsa_cache["literal_listener"]:
        rsa_cache["literal_listener"][utterance] = {}
    if scope not in rsa_cache["literal_listener"][utterance]:
        rsa_cache["literal_listener"][utterance][scope] = {}
    if qud not in rsa_cache["literal_listener"][utterance][scope]:
        def model():
            state = state_prior()
            q_state = qud_fun(qud, state)
            lit_meaning = meaning(utterance, state, scope)
            condition("lit_cond", lit_meaning)
            return state_to_tensor(q_state)

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["literal_listener"][utterance][scope][qud] = marginal

    return rsa_cache["literal_listener"][utterance][scope][qud]


# literal speaker

def literal_speaker(scope, state, qud, alpha=1., num_samples=10):
    if scope not in rsa_cache["literal_speaker"]:
        rsa_cache["literal_speaker"][scope] = {}
    if state not in rsa_cache["literal_speaker"][scope]:
        rsa_cache["literal_speaker"][scope][state] = {}
    if qud not in rsa_cache["literal_speaker"][scope][state]:
        def model():
            utterance = utterances_prior()
            q_state = qud_fun(qud, state)
            alpha_t = tensor(alpha).float()
            lit_lis = literal_listener(utterance, scope, qud, num_samples=num_samples)
            state_prob = lit_lis.log_prob(state_to_tensor(q_state))
            pyro.factor("speak_factor", alpha_t * (state_prob - cost(utterance)))
            return utterance_to_tensor(utterance)

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["literal_speaker"][scope][state][qud] = marginal
    return rsa_cache["literal_speaker"][scope][state][qud]


# Pragmatic listener (L1)
def pragmatic_listener(utterance, num_samples=10):
    if utterance not in rsa_cache["pragmatic_listener"]:
        def model():
            # //////// priors ////////
            state = state_prior()
            scope = scope_prior()
            qud = quds_prior()
            # ////////////////////////
            lit_speak = literal_speaker(scope, state, qud, num_samples=num_samples)
            pyro.sample("speak_utt", lit_speak, obs=utterance_to_tensor(utterance))
            state_t = state_to_tensor(state)
            scope_t = scope_to_tensor(scope)
            # order ?, dict ?
            result = torch.stack([state_t, scope_t])
            return result

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["pragmatic_listener"][utterance] = marginal
    return rsa_cache["pragmatic_listener"][utterance]

# Pragmatic speaker (S2)
def pragmatic_speaker(state, alpha=1., num_samples=10):
    if state not in rsa_cache["pragmatic_speaker"]:
        def model():
            utterance = utterances_prior()
            prag_lis = pragmatic_listener(utterance)
            state_prob = prag_lis.log_prob(state_to_tensor(state))
            pyro.factor("prag_speak_factor", state_prob)
            return utterance_to_tensor(utterance)

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["pragmatic_speaker"][state] = marginal
    return rsa_cache["pragmatic_speaker"][state]


prag_speak = pragmatic_speaker(2, num_samples=1_000)


for _ in range(100):
    tens = prag_speak.sample()
    print(tensor_to_utterances(tens))
