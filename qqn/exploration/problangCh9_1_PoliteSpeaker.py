from pyroapi import pyro
import pyro.distributions as dist
import torch
from torch import tensor

states = [1, 2, 3, 4, 5]
utterances = ["terrible", "bad", "okay", "good", "amazing"]

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
def meaning(utterance, state):
    return dist.Bernoulli(literal_semantics[utterance][state - 1])

# value function scales social utility by a parameter lambda
lam = 1.25 # value taken from MAP estimate from Yoon, Tessler, et al. 2016
def value_function(s):
  return lam * s


def literal_listener(utterance, num_samples=10):
    def model():
        state = dist.Categorical(logits=tensor(states)).sample()
        m = meaning(utterance, state)
        condition(m)
        return state

    importance = pyro.infer.Importance(model, num_samples=num_samples)
    importance.run()
    marginal = pyro.infer.EmpiricalMarginal(importance)
    return marginal


def speaker1(state, phi, alpha=10., num_samples=10):
    def model():
        utterance = dist.Categorical(logits=torch.zeros(len(utterances))).sample()
        L0_posterior = literal_listener(utterance)

        util = dict(
          epistemic = L0_posterior.prob_log(state),
            social = expectation(L0_posterior, value_function())
        )
        speak_util = phi * util["epistemic"] + (1 - phi) * utility["social"]
        alpha_t = tensor(alpha).float()

        pyro.factor("speak_factor", alpha_t * speak_util)
        return utterance

    importance = pyro.infer.Importance(model, num_samples=num_samples)
    importance.run()
    marginal = pyro.infer.EmpiricalMarginal(importance)
    return marginal


speaker1(1, 0.99)

