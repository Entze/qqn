import pyro.distributions as dist
from torch import tensor

from qqn.exploration.webppl import viz


def alt_beta(g, d):
    a = g * d;
    b = (1 - g) * d;
    return dist.Beta(a, b)


def flip(p=0.5):
    return dist.Bernoulli(p).sample()


def fep_gen():
    return dict(
        kind="fep",
        wings=flip(0.5),
        legs=flip(0.01),
        claws=flip(0.01),
        height=alt_beta(0.5, 10).sample()
    )


def wug_gen():
    return dict(
        kind="wug",
        wings=flip(0.5),
        legs=flip(0.99),
        claws=flip(0.3),
        height=alt_beta(0.2, 10).sample()
    )


def glippet_gen():
    return dict(
        kind="glippet",
        wings=flip(0.5),
        legs=flip(0.99),
        claws=flip(0.2),
        height=alt_beta(0.8, 10).sample())


state_bins = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]


def make_histogram(prevalences):
    return [
        sum([1 for p in prevalences if p == b], 0.001)
        for b in state_bins]


the_world = [fep_gen() for _ in range(10)] + [wug_gen() for _ in range(10)] + [glippet_gen() for _ in range(10)]

kinds = set(c["kind"] for c in the_world)
print(kinds)

print('height distribution over all creatures')
print([c["height"] for c in the_world])

print('')


def prevalence(world, kind, property):
    selection = [c for c in world if c['kind'] == kind]
    properties = tensor([c[property] for c in selection])
    return max(0.01, round(properties.mean().item(), 1))


def prevalence_prior(property, world):
    p = [prevalence(world, k, property) for k in kinds]
    return make_histogram(p)


print([c["legs"] for c in the_world])
viz(prevalence_prior("legs", the_world))