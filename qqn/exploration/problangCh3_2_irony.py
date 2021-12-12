import pyroapi
from frozenlist import FrozenList
from pyroapi import pyro
import pyro.distributions as dist
import torch
from torch import tensor

rsa_cache = dict(literal_listener={}, pragmatic_speaker={}, pragmatic_listener={})


def condition(name: str, val: bool = False):
    pyro.factor(name, tensor(float(0 if val else '-inf')))


# There are three possible states the weather could be in:
# terrible, ok, or amazing

states = ['terrible', 'ok', 'amazing']

# Since we are in California, the prior over these states
# are the following. Once could also imagine this being
# the prior in a certain context, e.g. when it's clearly
# sunny and nice out.

state_dist = dist.Categorical(logits=tensor([.1, .50, .50]))


def tensor_to_state(tens):
    return states[int(tens.item())]


def state_prior():
    sample_t = state_dist.sample()
    return tensor_to_state(sample_t)


def naive_state_prior():
    sample_t = dist.Categorical(logits=torch.zeros(len(states))).sample()
    return tensor_to_state(sample_t)


# Valence prior defined in terms of negative valence.
# If the current state is terrible, it's extremely likely
# that the valence associated is negative. If it's ok, then
# the valence could be negative or positive with equal
# probability.

valence_dict = dict(
    terrible=0.99,
    ok=0.5,
    amazing=0.01
)

valences = list(valence_dict.keys())


def valence_prior(state):
    prob = valence_dict[state]
    b = bool(dist.Bernoulli(prob).sample().item())
    return -1 if b else 1


# Define binary arousals (could model as continuous).
arousals = ["low", "high"]

# Define goals and goal priors. Could want to communicate state of the world,
# valence about it, or arousal (intensity of feeling) about it.
goals = ["goalState", "goalValence", "goalArousal"]

goal_dist = dist.Categorical(logits=torch.zeros(len(goals)))


def tensor_to_goal(tens):
    return goals[int(tens.item())]


def goal_prior():
    sample_t = goal_dist.sample()
    return tensor_to_goal(sample_t)


# Assume possible utterances are identical to possible states
utterances = states.copy()

utterances_dist = dist.Categorical(logits=torch.zeros(len(utterances)))


def tensor_to_utterance(tens):
    return tensor_to_state(tens)


def utterance_to_tensor(utt):
    for i, u in enumerate(utterances):
        if u == utt:
            return tensor(i).float()


# Assume cost of utterances is uniform.
def utterances_prior():
    sample_t = utterances_dist.sample()
    return tensor_to_utterance(sample_t)


# Sample arousal given a state.

arousal_dict = dict(
    terrible=[.1, .9],
    ok=[.9, .1],
    amazing=[.1, .9]
)


def tensor_to_arousal(tens):
    return arousals[int(tens.item())]


def arousal_to_tensor(arousal):
    for i, a in enumerate(arousals):
        if arousal == a:
            return tensor(i).float()


def state_to_tensor(state):
    for i, s in enumerate(states):
        if state == s:
            return tensor(i).float()


def valence_to_tensor(valence):
    return tensor(valence).float()


def arousal_prior(state):
    logits = arousal_dict[state]
    sample_t = dist.Categorical(logits=tensor(logits)).sample()
    return tensor_to_arousal(sample_t)


# Literal interpretation is just whether utterance equals state
def literal_interpretation(utterance, state):
    return utterance == state


# A speaker's goal is satisfied if the listener infers the correct
# and relevant information.
def goal_state(goal, state, valence, arousal):
    if goal == "goalState":
        return state
    elif goal == "goalValence":
        return valence
    elif goal == "goalArousal":
        return arousal
    assert False


def goal_state_t(goal, state, valence, arousal):
    if goal == "goalState":
        return state_to_tensor(state)
    elif goal == "goalValence":
        return valence_to_tensor(valence)
    elif goal == "goalArousal":
        return arousal_to_tensor(arousal)
    assert False


# Define a literal listener

def literal_listener(utt, goal, num_samples=10):
    if utt not in rsa_cache["literal_listener"]:
        rsa_cache["literal_listener"][utt] = {}
    if goal not in rsa_cache["literal_listener"][utt]:
        def model():
            state = naive_state_prior()
            valence = valence_prior(state)
            arousal = arousal_prior(state)
            lit_interpret = literal_interpretation(utt, state)
            condition("lit_cond", lit_interpret)
            return goal_state_t(goal, state, valence, arousal)

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["literal_listener"][utt][goal] = marginal

    return rsa_cache["literal_listener"][utt][goal]


# Define a speaker

def pragmatic_speaker(state, valence, arousal, goal, alpha=1., num_samples=10):
    if state not in rsa_cache["pragmatic_speaker"]:
        rsa_cache["pragmatic_speaker"][state] = {}
    if valence not in rsa_cache["pragmatic_speaker"][state]:
        rsa_cache["pragmatic_speaker"][state][valence] = {}
    if arousal not in rsa_cache["pragmatic_speaker"][state][valence]:
        rsa_cache["pragmatic_speaker"][state][valence][arousal] = {}
    if goal not in rsa_cache["pragmatic_speaker"][state][valence][arousal]:
        def model():
            utterance = utterances_prior()
            alpha_t = tensor(alpha).float()
            lit_lis = literal_listener(utterance, goal, num_samples=num_samples)
            state_prob = lit_lis.log_prob(goal_state_t(goal, state, valence, arousal))

            pyro.factor("speak_factor", alpha_t * state_prob)
            return utterance_to_tensor(utterance)

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["pragmatic_speaker"][state][valence][arousal][goal] = marginal
    return rsa_cache["pragmatic_speaker"][state][valence][arousal][goal]


# Define a pragmatic listener

def pragmatic_listener(utterance, num_samples=10):
    if utterance not in rsa_cache["pragmatic_listener"]:
        def model():
            # //////// priors ////////
            state = state_prior()
            valence = valence_prior(state)
            arousal = arousal_prior(state)
            goal = goal_prior()
            # ////////////////////////

            prag_speak = pragmatic_speaker(state, valence, arousal, goal, num_samples=num_samples)
            pyro.sample("speak_utt", prag_speak, obs=utterance_to_tensor(utterance))
            state_t = state_to_tensor(state)
            valence_t = valence_to_tensor(valence)
            arousal_t = arousal_to_tensor(arousal)
            result = torch.stack([state_t, valence_t, arousal_t])
            return result

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["pragmatic_listener"][utterance] = marginal
    return rsa_cache["pragmatic_listener"][utterance]


prag_list = pragmatic_listener("terrible", num_samples=1_000)

for _ in range(100):
    tens = prag_list.sample()
    l = tens.tolist()
    print(tensor_to_state(tensor(l[0])), l[1], tensor_to_arousal(tensor(l[2])))
