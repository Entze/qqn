import random

import torch
from pyroapi import pyro
import torch.distributions as torchdist
from torch import tensor

from qqn.exploration.RSA_POC3 import knockout

cache = {
    "literal_listener": {
    },
    "speaker": {

    },
    "pragmatic_listener": {

    }
}

config = {
    "num_samples": 1_000
}


def lexical_meaning(word, world):
    word_meanings = {

        "blond": {
            "sem": lambda obj: obj['blond'],
            "syn": {"dir": 'L', "int": 'NP', "out": 'S'}},

        "nice": {
            "sem": lambda obj: obj['nice'],
            "syn": {"dir": 'L', "int": 'NP', "out": 'S'}},

        "Bob": {
            "sem": list(filter(lambda obj: obj["name"] == "Bob", world))[0],
            "syn": 'NP'},

        "some": {
            "sem": lambda p: lambda q: len(list(filter(q, filter(p, world)))) > 0,
            "syn": {"dir": 'R',
                    "int": {"dir": 'L', "int": 'NP', "out": 'S'},
                    "out": {"dir": 'R',
                            "int": {"dir": 'L', "int": 'NP', "out": 'S'},
                            "out": 'S'}}},

        "all": {
            "sem": lambda p: lambda q: len(list(filter(neg(q), filter(p, world)))) == 0,
            "syn": {"dir": 'R',
                    "int": {"dir": 'L', "int": 'NP', "out": 'S'},
                    "out": {"dir": 'R',
                            "int": {"dir": 'L', "int": 'NP', "out": 'S'},
                            "out": 'S'}}}}

    return word_meanings.get(word, {"sem": None, "syn": ''})


def neg(q):
    return lambda x: not q(x)


def combine_meaning(meanings):
    assert meanings, "meanings must not be empty"
    possible_comb = can_apply(meanings, 0)
    print(possible_comb)
    assert possible_comb, "possible_comb must not be empty"
    i = random.choice(possible_comb)
    s = meanings[i]["syn"]
    f = meanings[i]["sem"]

    if s['dir'] == 'L':
        a = meanings[i - 1]["sem"]
        new_meaning = dict(sem=f(a), syn=s['out'])
        ret = meanings[0:i - 1] + [new_meaning] + meanings[i + 1:]
        return ret

    elif s['dir'] == 'R':
        a = meanings[i + 1]["sem"]
        new_meaning = dict(sem=f(a), syn=s['out'])
        ret = meanings[0:i] + [new_meaning] + meanings[i + 2:]
        return ret

    else:
        assert False, "Direction has to be in {L,R} but is" + s['dir']
    before_len = len(meanings)
    print(possible_comb)
    print("Inserting at", insert_before, "Removing from", insert_before + 1)
    meanings.insert(insert_before, {"sem": f(a), "syn": s['out']})
    meanings.pop(insert_before + 1)
    assert len(meanings) == before_len, "The size of meanings may never change"
    return meanings


def can_apply(meanings, i):
    if i == len(meanings):
        return []
    s = meanings[i]["syn"]
    if 'dir' in s:
        a = (s['dir'] == 'L' and syntax_match(s["int"], meanings[i - 1]["syn"])) or \
            (s['dir'] == 'R' and syntax_match(s["int"], meanings[i + 1]["syn"]))
        # a = (syntax_match(s["int"], meanings[i - 1]["syn"]) if s['dir'] == 'L' else False) or \
        #    (syntax_match(s["int"], meanings[i + 1]["syn"]) if s['dir'] == 'R' else False)
        if a:
            return [i] + can_apply(meanings, i + 1)
    return can_apply(meanings, i + 1)


def syntax_match(s, t):
    if 'dir' in s and 'dir' in t:
        return s["dir"] == t["dir"] and syntax_match(s["int"], t["int"]) and syntax_match(s["out"], t["out"])
    return s == t


def combine_meanings(meanings):
    if len(meanings) == 1:
        return meanings[0]["sem"]
    return combine_meanings(combine_meaning(meanings))


########################################################################################################################

name_dict = dict(
    Bob=0,
    Bill=1,
    Alice=2,
)

name_list = list(name_dict.keys())


def object_to_tensor(obj):
    return tensor([float(name_dict[obj["name"]]), obj["blond"], obj["nice"]])


def tensor_to_object(tens):
    name = tens[0]
    blond = tens[1]
    nice = tens[2]
    return dict(name=name_list[int(name)], blond=int(blond), nice=int(nice))


def world_to_tensor(world):
    tensor_l = [object_to_tensor(obj) for obj in world]
    return torch.stack(tensor_l)


def tensor_to_world(tens):
    obj_list = tens.tolist()
    return [tensor_to_object(obj) for obj in obj_list]


def flip(prior=0.5):
    return torchdist.Bernoulli(prior).sample()


def make_object(name):
    return dict(name=name, blond=flip(), nice=flip())


def world_generation():
    return [make_object(name) for name in ["Bob", "Bill", "Alice"]]


def meaning(utterance: str, world):
    return combine_meanings(
        [m for m in [lexical_meaning(w, world) for w in utterance.split(' ')] if "sem" in m and m["sem"] is not None]
    )


def basic_evidence(utterance):
    def model():
        world = world_generation()
        m = meaning(utterance, world)
        knockout("Basic Evidence", m)
        return world_to_tensor(world)

    return model


def literal_listener(utterance):
    """
    A distribution over worlds, with respect to a specific utterance.
    :return: an Infer object (Empirical Distribution)
    """
    if utterance not in cache["literal_listener"]:
        importance = pyro.infer.Importance(model=basic_evidence(utterance), num_samples=config["num_samples"])
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        cache["literal_listener"][utterance] = marginal

    return cache["literal_listener"][utterance]


examples = []
for _ in range(1000):
    s = literal_listener("all blond people are nice").sample()
    w = tensor_to_world(s)
    assert all(not bool(o['blond']) or o['nice'] for o in w)
    print(w)
    examples.append(s)

# viz(tensor(exsamples))
