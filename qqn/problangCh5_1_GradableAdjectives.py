import pyroapi
from frozenlist import FrozenList
from pyroapi import pyro
import pyro.distributions as dist
import torch
from torch import tensor

rsa_cache = dict(literal_listener={}, pragmatic_speaker={}, pragmatic_listener={})


def condition(name: str, val: bool = False):
    pyro.factor(name, tensor(float(0 if val else '-inf')))


book = {
    "prices": [2, 6, 10, 14, 18, 22, 26, 30],
    "probabilities": [1, 2, 3, 4, 4, 3, 2, 1]
}

book_price_dist = pyro.Categorical(logits=tensor(book["probabilities"]))


def tensor_to_book_price(tens):
    return book["prices"][int(tens.item())]



def book_price_prior():
    sample_t = book_price_dist.sample()
    return tensor_to_book_price(sample_t)


def theta_prior():
    return pyro.Categorical(logits=torch.zeros(len(book["prices"]))).sample()


utterances = ["expensive", ""]

cost = {
    "expensive": 1,
    "": 0
}


def tensor_to_utterance(tens):
    return utterances[int(tens.item())]

def utterance_prior():
    return pyro.Categorical(logits=torch.zeros(len(utterances))).sample()


def meaning(utterance, price, theta):
    if utterance == "expensive":
        return price >= theta
    return True


# Define a literal listener

def literal_listener(utt, theta, num_samples=10):
    if utt not in rsa_cache["literal_listener"]:
        rsa_cache["literal_listener"][utt] = {}
    if theta not in rsa_cache["literal_listener"][utt]:
        def model():
            price = book_price_dist.sample()
            lit_meaning = meaning(utt, price, theta)
            condition("lit_cond", lit_meaning)
            return price

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["literal_listener"][utt][theta] = marginal

    return rsa_cache["literal_listener"][utt][theta]


# literal speaker

def literal_speaker(price, theta, alpha=1., num_samples=10):
    if price not in rsa_cache["literal_speaker"]:
        rsa_cache["literal_speaker"][price] = {}
    if theta not in rsa_cache["literal_speaker"][price]:
        def model():
            utterance = utterance_prior()
            alpha_t = tensor(alpha).float()
            lit_lis = literal_listener(utterance, theta, num_samples=num_samples)
            # price here price_t
            state_prob = lit_lis.log_prob(price)
            # not sure about tensor to utterance
            pyro.factor("speak_factor", alpha_t * (state_prob - cost[tensor_to_utterance(utterance)]))
            return utterance

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["literal_speaker"][price][theta] = marginal
    return rsa_cache["literal_speaker"][price][theta]


# Define a pragmatic listener

def pragmatic_listener(utterance, num_samples=10):
    if utterance not in rsa_cache["pragmatic_listener"]:
        def model():
            # //////// priors ////////
            price = book_price_dist.sample()
            theta = theta_prior()
            # ////////////////////////

            prag_speak = pragmatic_speaker(price, theta, num_samples=num_samples)
            pyro.sample("speak_utt", prag_speak, obs=utterance)
            result = torch.stack([price, theta])
            return result

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["pragmatic_listener"][utterance] = marginal
    return rsa_cache["pragmatic_listener"][utterance]


prag_list = pragmatic_listener("expensive", num_samples=1_000)

for _ in range(100):
    tens = prag_list.sample()
    l = tens.tolist()
    print(tensor_to_state(tensor(l[0])), l[1], tensor_to_arousal(tensor(l[2])))
