from pyroapi import pyro
import pyro
import pyro.distributions as dist
import torch
from torch import tensor
from qqn.initial_exploration.webppl import expectation

states = [1, 2, 3, 4, 5]
utterances = ["terrible", "bad", "okay", "good", "amazing"]

torch.manual_seed(0)
pyro.set_rng_seed(0)

# correspondence of utterances to states (empirically measured)
literal_semantics = {
    "terrible": [.95, .85, .02, .02, .02],
    "bad": [.85, .95, .02, .02, .02],
    "okay": [0.02, 0.25, 0.95, .65, .35],
    "good": [.02, .05, .55, .95, .93],
    "amazing": [.02, .02, .02, .65, 0.95]
}


# determine whether the utterance describes the state
# by flipping a coin with the literalSemantics weight
# ... state - 1 because of 0-indexing
def meaning(utterance_t, state_t):
    utterance = utterances[int(utterance_t.item())]
    state = states[int(state_t.item())]
    prob = literal_semantics[utterance][state - 1]
    return dist.Bernoulli(prob).sample()


# value function scales social utility by a parameter lambda
lam = 1.25  # value taken from MAP estimate from Yoon, Tessler, et al. 2016


def value_function(s):
    return lam * s


def condition(name: str, val: bool = False):
    pyro.factor(name, tensor(float(0 if val else '-inf')))


def literal_listener(utterance, num_samples=10):
    def model():
        state = dist.Categorical(logits=torch.zeros(len(states))).sample()
        m = meaning(utterance, state)
        condition("lit_cond", m)
        return state.float()

    importance = pyro.infer.Importance(model, num_samples=num_samples)
    importance.run()
    marginal = pyro.infer.EmpiricalMarginal(importance)
    return marginal


def speaker1(state, phi, alpha=10., num_samples=10):
    def model():
        utterance = dist.Categorical(logits=torch.zeros(len(utterances))).sample()
        L0_posterior = literal_listener(utterance, num_samples=num_samples)

        util = dict(
            epistemic=L0_posterior.log_prob(state),
            social=value_function(expectation(L0_posterior))
        )
        speak_util = phi * util["epistemic"] + (1 - phi) * util["social"]
        alpha_t = tensor(alpha).float()

        pyro.factor("speak_factor", alpha_t * speak_util)
        return utterance

    importance = pyro.infer.Importance(model, num_samples=num_samples)
    importance.run()
    marginal = pyro.infer.EmpiricalMarginal(importance)
    return marginal


speaker1(tensor(0.), tensor(0.99), num_samples=100)


def pragmatic_listener(utterance, num_samples=10):
    def model():
        state = dist.Categorical(logits=torch.zeros(len(states))).sample()
        r = list(range(5, 950, 5))
        l = tensor(r) / 100.
        phi_sample = dist.Categorical(logits=torch.zeros(len(r))).sample()
        phi = l[int(phi_sample.item())]
        S1 = speaker1(state, phi, num_samples=num_samples)
        pyro.sample("prag_lit", S1, obs=utterance)
        return torch.stack([state, phi])

    importance = pyro.infer.Importance(model, num_samples=num_samples)
    importance.run()
    marginal = pyro.infer.EmpiricalMarginal(importance)
    return marginal


listenerPosterior = pragmatic_listener(tensor(3.), num_samples=15)
# note in this case that visualizing the joint distribution via viz()
# produces the wrong joint distribution. this is a bug in the viz() program.
# we visualize the marginal distributions instead:


