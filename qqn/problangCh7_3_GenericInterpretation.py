from collections import defaultdict

import torch
from torch import tensor, Tensor
import pyro
import pyro.infer
import pyro.distributions as dist

from qqn.webppl import viz, viz_heatmap

rsa_cache = dict(literal_listener={}, literal_speaker={}, pragmatic_listener={}, discrete_beta={})

literal_meanings = dict(
    some=lambda prevalence: prevalence > 0.,
    most=lambda prevalence: prevalence > 0.5,
    all=lambda prevalence: prevalence == 1.,
    generic=lambda prevalence, theta: prevalence > theta
)

meaning_func = literal_meanings["generic"]
print(meaning_func(tensor(0.6), tensor(0.5)))

example_theta = 0.49

percentage_of_birds_that_lay_eggs = 0.5
percentage_of_birds_that_are_female = 0.5
percentage_of_mosquitos_that_carry_malaria = 0.02

print("Birds lay eggs?", meaning_func(percentage_of_birds_that_lay_eggs, example_theta))
print("Birds are female?", meaning_func(percentage_of_birds_that_are_female, example_theta))
print("Mosquitos carry malaria?", meaning_func(percentage_of_mosquitos_that_carry_malaria, example_theta))

all_kinds = [
    {"kind": "dog", "family": "mammal"},
    {"kind": "falcon", "family": "bird"},
    {"kind": "cat", "family": "mammal"},
    {"kind": "gorilla", "family": "mammal"},
    {"kind": "robin", "family": "bird"},
    {"kind": "alligator", "family": "reptile"},
    {"kind": "giraffe", "family": "mammal"},
]

all_kinds_dist = dist.Categorical(logits=torch.zeros(len(all_kinds)))

all_kinds_prevalence = defaultdict(float, dict(
    bird=0.5,
    reptile=0.2
))


def prevalence_prior():
    k_t = all_kinds_dist.sample()
    k = all_kinds[int(k_t.item())]
    prevalence = all_kinds_prevalence[k["family"]]
    return tensor(prevalence)


def condition(name: str, val: bool = False):
    pyro.factor(name, tensor(float(0 if val else '-inf')))


print(prevalence_prior())


def threshold_prior():
    sample_t = dist.Categorical(logits=torch.zeros(len(bins))).sample()
    return bins[int(sample_t.item())]


utterances = ['generic', 'silence']


def meaning(utt_t, prevalence: Tensor, threshold: Tensor):
    utt = utterances[int(utt_t.item())]
    if utt == 'generic':
        return torch.all(prevalence > threshold)
    return True


def literal_listener(utt, state_prior, num_samples=10):
    def model():
        prevalence = state_prior.sample()
        threshold = threshold_prior()
        m = meaning(utt, prevalence, threshold)
        condition("lit_cond", m)
        return prevalence

    importance = pyro.infer.Importance(model, num_samples=num_samples)
    importance.run()
    marginal = pyro.infer.EmpiricalMarginal(importance)
    return marginal


bins = list(map(lambda x:
                round(x / 100, 2), range(1, 100, 2)))


# function returns a discretized Beta distribution
def discrete_beta(g, d):
    if g not in rsa_cache['discrete_beta']:
        rsa_cache['discrete_beta'][g] = {}
    if d not in rsa_cache['discrete_beta'][g]:
        a = g * d
        b = (1 - g) * d

        def beta_pdf(x):
            return (x ** (a - 1)) * ((1 - x) ** (b - 1))

        probs = list(map(beta_pdf, bins))
        rsa_cache['discrete_beta'][g][d] = dist.Categorical(logits=tensor(probs))
    return rsa_cache['discrete_beta'][g][d]


print("prevalence prior for transient causes:")
viz(discrete_beta(0.01, 100))


def prior_model(**kwargs):
    def model():
        phi = kwargs["potential"]
        gamma = kwargs["prevalence_when_present"]
        delta = kwargs["concentration_when_present"]

        stable_dist = discrete_beta(gamma, delta)
        unstable_dist = discrete_beta(0.01, 100)

        pre = dist.Bernoulli(phi).sample()
        if pre == 1:
            prevalence_t = stable_dist.sample()
        else:
            prevalence_t = unstable_dist.sample()

        prevalence = bins[int(prevalence_t.item())]

        return tensor(prevalence)

    importance = pyro.infer.Importance(model, num_samples=kwargs.get("num_samples", 10))
    importance.run()
    marginal = pyro.infer.EmpiricalMarginal(importance)
    return marginal


prior = prior_model(potential=0.3, prevalence_when_present=0.5, concentration_when_present=10, num_samples=10_000)

#
# viz(prior_model(potential=0.3, prevalence_when_present=0.5, concentration_when_present=10, num_samples=10_000))
#
# lit_lis = literal_listener(tensor(0.0), num_samples=100)
#
# viz(lit_lis)



def utterance_prior():
    sample_t = dist.Categorical(logits=torch.zeros(len(utterances))).sample()
    return sample_t


def literal_speaker(prevalence, state_prior, alpha=1., num_samples=10):
    def model():
        utterance = utterance_prior()
        alpha_t = tensor(alpha).float()
        lit_lis = literal_listener(utterance, state_prior, num_samples=num_samples)
        state_prob = lit_lis.log_prob(prevalence)
        pyro.factor("speak_factor", alpha_t * (state_prob - 1))
        return utterance

    importance = pyro.infer.Importance(model, num_samples=num_samples)
    importance.run()
    marginal = pyro.infer.EmpiricalMarginal(importance)
    return marginal


viz(prior)

speak = literal_speaker(tensor(0.03), prior, num_samples=100)

print(speak.sample())
print(speak.sample())
print(speak.sample())

viz(speak)
