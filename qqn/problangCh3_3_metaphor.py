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




# Utterances


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


#prag_list = pragmatic_listener("terrible", num_samples=1_000)

'''
for _ in range(100):
    tens = prag_list.sample()
    l = tens.tolist()
    print(tensor_to_state(tensor(l[0])), l[1], tensor_to_arousal(tensor(l[2])))
'''


# John could either be a whale or a person.
categories = ["whale", "person"]

# It is extremely unlikely that John is actually a whale.
categories_dist = dist.Categorical(logits=tensor([.01,.99]))

def tensor_to_category(tens):
    return categories[int(tens.item())]

def categories_prior():
    sample_t = categories_dist.sample()
    return tensor_to_category(sample_t)

# Utterances
# The speaker could either say "John is a whale" or "John is a person."
utterances = ["whale", "person"]

# The utterances are equally costly.
utterances_dist = dist.Categorical(logits=tensor([1,1]))


def utterance_prior():
    sample_t = categories_dist.sample()
    return tensor_to_utterance(sample_t)


# The features of John being considered are "large", "graceful",
# "majestic." Features are binary.
feature_sets = [
    dict( large = l, graceful = g, majestic = m)
    for l in [0,1] for g in [0,1] for m in [0,1]
]


# information about feature priors (probabilistic world knowledge)
# obtained by an experimental study (see paper)
feature_set_dist = dict(
    whale = [0.30592786494628, 0.138078454222818,
                                      0.179114768847673, 0.13098781834847,
                                      0.0947267162507846, 0.0531420411185539,
                                      0.0601520520596695, 0.0378702842057509],
    paerson = [0.11687632453038, 0.105787535267869,
                                       0.11568145784997, 0.130847056136141,
                                       0.15288225956497, 0.128098151176801,
                                       0.114694702836614, 0.135132512637255])


def tensor_to_feature_set(tens):
    return feature_sets[int(tens.item())]


def feature_set_prior(category):
    logits = feature_set_dist[category]
    sample_t = dist.Categorical(logits=tensor(logits)).sample()
    return tensor_to_feature_set(sample_t)


# Speaker's possible goals are to communicate feature 1, 2, or 3
goals = ["large", "graceful", "majestic"]

# Prior probability of speaker's goal is set to uniform but can
# change with context/QUD.

goal_dist = dist.Categorical(logits=tensor([1,1,1]))


def tensor_to_goal(tens):
    return goals[int(tens.item())]


def goal_prior():
    sample_t = goal_dist.sample()
    return tensor_to_goal(sample_t)



# Speaker optimality parameter
alpha = 3

# Check if interpreted category is identical to utterance
def literal_interpretation(utterance, category):
    return utterance == category

# Check if goal is satisfied
def goal_state(goal, feature_set):
    if goal == "large":
        return feature_set["large"]
    elif goal == "graceful":
        return feature_set["graceful"]
    elif goal == "majestic":
        return feature_set["majestic"]
    assert False


def feature_set_to_tensor(param):
    pass


def goal_state_t(goal, feature_set):
    if goal == "large":
        return feature_set_to_tensor(feature_set["large"])
    elif goal == "graceful":
        return feature_set_to_tensor(feature_set["graceful"])
    elif goal == "majestic":
        return feature_set_to_tensor(feature_set["majestic"])
    assert False


#  Define a literal listener

def literal_listener(utt, goal, num_samples=10):
    if utt not in rsa_cache["literal_listener"]:
        rsa_cache["literal_listener"][utt] = {}
    if goal not in rsa_cache["literal_listener"][utt]:
        def model():
            category = dist.Categorical(logits=torch.zeros(len(categories)))
            feature_set = feature_set_prior(category)
            lit_interpret = literal_interpretation(utt, category)
            condition("lit_cond", lit_interpret)
            return goal_state_t(goal, feature_set)

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["literal_listener"][utt][goal] = marginal

    return rsa_cache["literal_listener"][utt][goal]


# Speaker model

def pragmatic_speaker(large, graceful, majestic, goal, alpha=1., num_samples=10):
    if large not in rsa_cache["pragmatic_speaker"]:
        rsa_cache["pragmatic_speaker"][large] = {}
    if graceful not in rsa_cache["pragmatic_speaker"][large]:
        rsa_cache["pragmatic_speaker"][large][graceful] = {}
    if majestic not in rsa_cache["pragmatic_speaker"][state][graceful]:
        rsa_cache["pragmatic_speaker"][large][graceful][majestic] = {}
    if goal not in rsa_cache["pragmatic_speaker"][large][graceful][majestic]:
        def model():
            utterance = utterances_prior()
            alpha_t = tensor(alpha).float()
            lit_lis = literal_listener(utterance, goal, num_samples=num_samples)
            state_prob = lit_lis.log_prob(goal_state_t(goal, dict(large = large, graceful = graceful, majestic = majestic)))

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
            category = category_prior()
            feature_set = feature_set_prior(category)
            large = feature_set["large"]
            graceful = feature_set["graceful"]
            majestic = feature_set["majestic"]
            goal = goal_prior()
            # ////////////////////////

            prag_speak = pragmatic_speaker(large, graceful, majestic goal, num_samples=num_samples)
            pyro.sample("speak_utt", prag_speak, obs=utterance_to_tensor(utterance))
            large_t = large_to_tensor(large)
            graceful_t = graceful_to_tensor(graceful)
            majestic_t = majestic_to_tensor(majestic)
            result = torch.stack([large_t, graceful_t, majestic_t])
            return result

        importance = pyro.infer.Importance(model, num_samples=num_samples)
        importance.run()
        marginal = pyro.infer.EmpiricalMarginal(importance)
        rsa_cache["pragmatic_listener"][utterance] = marginal
    return rsa_cache["pragmatic_listener"][utterance]


var pragmaticListener = function(utterance) {
  Infer({model: function() {
    var category = categoriesPrior()
    var featureSet = featureSetPrior(category)
    var large = featureSet.large
    var graceful = featureSet.graceful
    var majestic = featureSet.majestic
    var goal = goalPrior()
    observe(speaker(large, graceful, majestic, goal), utterance)
    return {category, large, graceful, majestic}
  }})
}


prag_list = pragmatic_listener("whale", num_samples=1_000)

'''
for _ in range(100):
    tens = prag_list.sample()
    l = tens.tolist()
    print(tensor_to_state(tensor(l[0])), l[1], tensor_to_arousal(tensor(l[2])))
'''
