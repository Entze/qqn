from frozenlist import FrozenList
import pyro
import pyro.infer
import pyro.distributions as dist
import torch
from torch import tensor, Tensor

from qqn.exploration.webppl import viz

torch.manual_seed(0)
pyro.set_rng_seed(0)

rsa_cache = dict(literal_listener={}, pragmatic_speaker={}, pragmatic_listener={})


def num_of_trues(state):
    if isinstance(state, list) or isinstance(state, FrozenList):
        if isinstance(state[0], Tensor):
            return sum(int(bool(a.item())) for a in state)
        return sum(int(bool(a)) for a in state)
    elif isinstance(state, Tensor):
        return state.cpu().apply_(lambda a: int(bool(a))).sum().item()


bool_val_list = [True, False]
states = [[a1, a2, a3] for a1 in bool_val_list for a2 in bool_val_list for a3 in bool_val_list]


# print(states)


# red apple base rate baserate = 0.8
# state builder
def state_prior(base_rate=0.8):
    bern_dist = dist.Bernoulli(base_rate)
    a1 = bern_dist.sample()
    a2 = bern_dist.sample()
    a3 = bern_dist.sample()
    return torch.stack([a1, a2, a3])


# speaker belief function
def belief(actual_state, access):
    uniform = dist.Categorical(logits=torch.zeros(3))

    def fun(access_val, state_val):
        state = state_prior()
        return tensor(float(state_val)) if access_val else state[int(uniform.sample().item())]

    l = FrozenList([fun(access[i], a) for i, a in enumerate(actual_state)])
    l.freeze()
    return l

    # return list(map(lambda z: fun(z[1], z[0]), zip(actual_state, access)))
    # return map2(fun, access, actualState)


utterances = ['all', 'some', 'none']
utt_dist = dist.Categorical(logits=torch.zeros(len(utterances)))


# possible utterances
def utterance_prior(name):
    return pyro.sample(name, utt_dist)


cost = dict(
    all=1,
    some=1,
    none=1)

# meaning function to interpret the utterances
literal_meanings = dict(
    all=lambda state: all(state),
    some=lambda state: any(state),
    none=lambda state: not any(state))

b_state = belief([True, True, True], [True, True, False])
# print(b_state)
# print(num_of_trues(b_state))
# print("1000 runs of the speaker's belief function:")
samples = [tensor(float(num_of_trues(belief([True, True, True], [True, True, False])))) for _ in range(3)]


# viz(torch.stack(samples))


def condition(name: str, val: bool = False):
    pyro.factor(name, tensor(float(0 if val else '-inf')))


# literal listener
def literal_listener(utt, num_samples=10):
    if utt not in rsa_cache["literal_listener"]:
        def model():
            state_t = state_prior()
            state = state_t.tolist()
            meaning = literal_meanings[utt]
            m = meaning(state)
            assert isinstance(m, bool)
            condition("lit_cond", m)
            return state_t

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["literal_listener"][utt] = marginal

    return rsa_cache["literal_listener"][utt]


def pragmatic_speaker(access, state, alpha=1., num_samples=10):
    if (access, state) not in rsa_cache["pragmatic_speaker"]:
        def model():
            utt_t = utterance_prior("prag_utt")
            utt = list(literal_meanings.keys())[int(utt_t.item())]
            alpha_t = tensor(alpha).float()
            b_state = [float(b) for b in belief(state, access)]
            b_state_t = tensor(b_state)
            lit_lis = literal_listener(utt, num_samples=num_samples)
            state_prob = lit_lis.log_prob(b_state_t)
            pyro.factor("speak_factor", alpha_t * (state_prob - tensor(cost.get(utt, 0))))
            return utt_t

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["pragmatic_speaker"][(access, state)] = marginal
    return rsa_cache["pragmatic_speaker"][(access, state)]


def utt_to_tensor(utt, utts):
    for k, v in enumerate(utts):
        if v == utt:
            return tensor(k).float()


# pragmatic listener
def pragmatic_listener(utt, access, num_samples=10):
    if utt not in rsa_cache["pragmatic_listener"]:
        def model():
            state_t = state_prior()
            state = FrozenList(state_t.tolist())
            state.freeze()
            utt_t = utt_to_tensor(utt, utterances)
            pyro.sample("speak_utt", pragmatic_speaker(access, state, num_samples=num_samples), obs=utt_t)
            return tensor(num_of_trues(state))

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["pragmatic_listener"][utt] = marginal
    return rsa_cache["pragmatic_listener"][utt]


l1 = FrozenList([True, True, True])
l1.freeze()

l2 = FrozenList([True, True, False])
l2.freeze()

print("pragmatic listener for a full-access speaker:")
prag_lis = pragmatic_listener('some', l1, num_samples=10000)
viz(prag_lis)
print("pragmatic listener for a partial-access speaker:")
viz(pragmatic_listener('some', l2, num_samples=10000))
