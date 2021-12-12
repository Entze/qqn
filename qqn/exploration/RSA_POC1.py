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


def world_generation(number_of_people=3):
    """
    Draws from a distribution of worlds.
    :return: a concrete world, with type dict
    """
    world = {
        'type': 'world',
        'number_of_people': number_of_people,
        'number_of_nice_people': torchdist.Categorical(
            logits=torch.zeros(number_of_people)).sample().item()
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
        return world

    return model


def literal_listener(utterance):
    """
    A distribution over worlds, with respect to a specific utterance.
    :return: an Infer object (Empirical Distribution)
    """
    importance = pyro.infer.Importance(model=basic_evidence(utterance))
    importance.run()
    marginal = pyro.infer.EmpiricalMarginal(importance)
    return marginal


def listenable_evidence(world):
    def model():
        utterance = utterance_generation()
        knockout("Listenable Evidence", world == pyro.sample("Basic Observation", literal_listener(utterance)))
        return utterance

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
    return Infer(model=listenable_evidence(world), posterior_method="forward")


def speakable_evidence(utterance):
    def model():
        world = world_generation()
        knockout("Speakable Evidence", world == pyro.sample("Basic Intent", speaker(world)))


def pragmatic_listener(utterance):
    """
    A distribution over worlds, with respect to the identification of a specific world with a given utterance by a
    speaker.
    :return: an Infer object (Empirical Distribution)
    """
    return Infer(model=speakable_evidence(utterance), posterior_method="forward")


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
        applicable = world['number_of_nice_people'] == world['number_of_people']
    elif utterance == none_people:
        applicable = world['number_of_nice_people'] == 0
    return applicable


literal_listener(some_people)
