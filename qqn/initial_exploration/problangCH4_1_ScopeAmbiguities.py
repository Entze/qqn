from pyroapi import pyro
import pyro.distributions as dist
import torch
from torch import tensor

# cache
rsa_cache = dict(literal_listener={}, pragmatic_speaker={}, pragmatic_listener={})


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
    for i, s in enumerate(state):
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
    sample_t = states_dist.sample()
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


# Literal listener (L0)

def literal_listener(utterance, scope, num_samples=10):
    if utterance not in rsa_cache["literal_listener"]:
        rsa_cache["literal_listener"][utterance] = {}
    if scope not in rsa_cache["literal_listener"][utterance]:
        def model():
            state = state_prior()
            lit_meaning = meaning(utterance, state, scope)
            condition("lit_cond", meaning(utterance, state, scope))
            return state_to_tensor(lit_meaning)

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["literal_listener"][utterance][scope] = marginal

    return rsa_cache["literal_listener"][utterance][scope]


# pragmatic speaker

def pragmatic_speaker(scope, state, alpha=1., num_samples=10):
    if scope not in rsa_cache["pragmatic_speaker"]:
        rsa_cache["pragmatic_speaker"][scope] = {}
    if state not in rsa_cache["pragmatic_speaker"][scope]:
        def model():
            utterance = utterances_prior()
            alpha_t = tensor(alpha).float()
            lit_lis = literal_listener(utterance, scope, num_samples=num_samples)
            state_prob = lit_lis.log_prob(state_to_tensor(state))
            pyro.factor("speak_factor", alpha_t * (state_prob - cost(utterance)))
            return utterance_to_tensor(utterance)

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["pragmatic_speaker"][scope][state] = marginal
    return rsa_cache["pragmatic_speaker"][scope][state]


# Pragmatic listener (L1)
def pragmatic_listener(utterance, num_samples=10):
    if utterance not in rsa_cache["pragmatic_listener"]:
        def model():
            # //////// priors ////////
            state = state_prior()
            scope = scope_prior()
            # ////////////////////////
            prag_speak = pragmatic_speaker(scope, state, num_samples=num_samples)
            pyro.sample("speak_utt", prag_speak, obs=utterance_to_tensor(utterance))
            state_t = state_to_tensor(state)
            scope_t = scope_to_tensor(scope)
            result = torch.stack([state_t, scope_t])    # order ?
            return result

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["pragmatic_listener"][utterance] = marginal
    return rsa_cache["pragmatic_listener"][utterance]

prag_list = pragmatic_listener("every-not", num_samples=1_000)

for _ in range(100):
    tens = prag_list.sample()
    l = tens.tolist()
    print(tensor_to_state(tensor(l[0])), l[1], tensor_to_arousal(tensor(l[2])))