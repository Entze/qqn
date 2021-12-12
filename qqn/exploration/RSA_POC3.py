import random

import pyro
import pyro.infer

import torch
import torch.distributions as torchdist
from torch import tensor

from webppl import Infer, viz

some_people = 'some of the people are nice'
all_people = 'all of the people are nice'
none_people = 'none of the people are nice'

utterances_dict = {
    some_people: 0,
    all_people: 1,
    none_people: 2
}

torch.manual_seed(0)
pyro.set_rng_seed(0)

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


def world_to_tensor(world):
    return tensor(float(world["number_of_nice_people"]))


def utterance_to_tensor(utterance):
    return tensor(float(utterances_dict[utterance]))


def world_generation(number_of_people=3):
    """
    Draws from a distribution of worlds.
    :return: a concrete world, with type dict
    """
    world = {
        'number_of_nice_people': torchdist.Categorical(
            logits=torch.zeros(number_of_people + 1)).sample().item()
    }

    return world


def utterance_generation():
    """
    Draws from a distribution of utterances.
    :return: a concrete utterance, with type dict, the utterance is in the key "utterance"
    """
    utterance = [some_people, all_people, none_people][torchdist.Categorical(logits=torch.zeros(3)).sample().item()]
    return utterance


def knockout(name, value: bool = False):
    pyro.factor(name, tensor(float(0 if value else '-inf')))


def basic_evidence(utterance):
    def model():
        world = world_generation()
        m = meaning(utterance, world)
        # pyro.factor("Basic Evidence", tensor(float(0 if m else '-inf')))
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


def listenable_evidence(world):
    def model():
        utterance = utterance_generation()
        observation = pyro.sample("Basic Observation", literal_listener(utterance))
        evidence = world_to_tensor(world) == observation
        knockout("Listenable Evidence", evidence)
        utterance_t = utterances_dict[utterance]
        return tensor(float(utterance_t))

    return model


def n_rank_evidence(evidence):
    if isinstance(evidence, str):
        # is utterance
        pass
    elif isinstance(evidence, dict):
        # is world
        pass
    else:
        raise Exception(f"Cannot proceed with evidence {evidence}")


def speaker(world):
    """
    A distribution over utterances, with respect to the identification of a specific world, with the utterance by a
    literal listener (literal_listener).
    :return: an Infer object (Empirical Distribution)
    """
    if world["number_of_nice_people"] not in cache["speaker"]:
        importance = pyro.infer.Importance(listenable_evidence(world), num_samples=config["num_samples"])
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        cache["speaker"][world["number_of_nice_people"]] = marginal
    return cache["speaker"][world["number_of_nice_people"]]


def speakable_evidence(utterance):
    def model():
        world = world_generation()
        knockout("Speakable Evidence", utterance_to_tensor(utterance) == pyro.sample("Basic Intent", speaker(world)))
        return world_to_tensor(world)

    return model


def pragmatic_listener(utterance):
    """
    A distribution over worlds, with respect to the identification of a specific world with a given utterance by a
    speaker.
    :return: an Infer object (Empirical Distribution)
    """
    if utterance not in cache["pragmatic_listener"]:
        importance = pyro.infer.Importance(model=speakable_evidence(utterance), num_samples=config["num_samples"])
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        cache["pragmatic_listener"][utterance] = marginal
    return cache["pragmatic_listener"][utterance]


def meaning(utterance, world):
    """
    Checks if an utterance is applicable in a given world.
    :param utterance:
    :param world:
    :return: true if the utterance is applicable otherwise false
    """
    applicable = True
    if utterance == some_people:
        applicable = world['number_of_nice_people'] > 0
    elif utterance == all_people:
        applicable = world['number_of_nice_people'] == 3
    elif utterance == none_people:
        applicable = world['number_of_nice_people'] == 0
    return applicable


# for utt in utterances_dict.keys():
#     literal_listener(utt)
# for n in range(4):
#     speaker({"number_of_nice_people": n})
# for utt in utterances_dict.keys():
#     pragmatic_listener(utt)
# for n1, rsa in cache.items():
#     for n2, m in rsa.items():
#         viz(m, title=n1 + " " + str(n2))
# viz(pragmatic_listener(some_people))
