import pyro
import torch
from pyro import sample
import pyro.distributions as dist
from pyro.infer import Trace_ELBO, config_enumerate, TraceEnum_ELBO, SVI, NUTS, MCMC
from torch import tensor, Tensor
from pyro.optim import Adam
from tqdm import trange

agent = {
    "goal": None,
    "strategy": {
        "rules": [],
        "strategy_heuristic": None,
        "tactics_heuristic": None,
    },
    "tactics": []
}

actions = ["italian", "french"]
pyro.clear_param_store()


def transition(state, action):
    if action == "italian":
        return "pizza"
    return "steak frites"


def meal_dist(action):
    probs = [
        float(0 in action),
        float(1 in action)
    ]
    return dist.Categorical(tensor(probs))


restaurants = {
    "options": [
        {
            "name": "italian",
            "preferences": ""
        },
        {
            "name": "french"
        },
        {
            "name": "turkish"
        }
    ]
}


def inference_agent_model():
    action = pyro.sample("action", dist.Categorical(tensor([1 / 3, 1 / 3, 1 / 3])))
    if action.item() == 0:
        meal = pyro.sample("meal", dist.Categorical(tensor([0.9, 0.1])), obs=tensor(0.))
    elif action.item() == 1:
        meal = pyro.sample("meal", dist.Categorical(tensor([0.5, 0.5])), obs=tensor(0.))
    else:
        meal = pyro.sample("meal", dist.Categorical(tensor([0.0, 1.0])), obs=tensor(0.))
    return action


def inference_agent_guide():
    italian_preference = pyro.param("italian_preference", tensor(1.))
    turkish_preference = pyro.param("turkish_preference", tensor(1.))
    french_preference = pyro.param("french_preference", tensor(1.))
    action = pyro.sample("action",
                         dist.Categorical(logits=torch.stack((italian_preference, turkish_preference, french_preference))))
    if action.item() == 0:
        meal = pyro.sample("meal", dist.Categorical(tensor([0.9, 0.1])), obs=tensor(0.))
    elif action.item() == 1:
        meal = pyro.sample("meal", dist.Categorical(tensor([0.5, 0.5])), obs=tensor(0.))
    else:
        meal = pyro.sample("meal", dist.Categorical(tensor([0.0, 1.0])), obs=tensor(0.))
    return action


# setup the optimizer
adam_params = {"lr": 0.005}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(inference_agent_model, inference_agent_guide, optimizer, loss=Trace_ELBO())

param_vals = []

# do gradient steps
for step in trange(100_000):
    svi.step()
    param_vals.append({k: pyro.param(k).exp().item() for k in ["italian_preference", "turkish_preference", "french_preference"]})

print(param_vals)

"""
def plot_posterior(mcmc):
    # get `transition_prob` samples from posterior
    trace_transition_prob = mcmc.get_samples()["transition_prob"]

    plt.figure(figsize=(10, 6))
    for i in range(2):
        for j in range(2):
            sns.distplot(trace_transition_prob[:, i, j], hist=False, kde_kws={"lw": 2},
                         label="transition_prob[{}, {}], true value = {:.2f}"
                         .format(i, j, transition_prob[i, j]))
    plt.xlabel("Probability", fontsize=13)
    plt.ylabel("Frequency", fontsize=13)
    plt.title("Transition probability posterior", fontsize=15)


def generate_traces(states, meals):
    with pyro.plate("prob_meal", 2):
        transition_prob = pyro.sample("transition_prob", dist.Dirichlet(torch.ones(2)))
        emission_prob = pyro.sample("emission_prob", dist.Dirichlet(torch.eye(2)))

    step = states[0]
    for t in range(len(meals)):
        if t > 0:
            step = pyro.sample("step_{}".format(t), dist.Categorical(transition_prob[step]))
        pyro.sample("meal_{}".format(t), dist.Categorical(emission_prob[step]), obs=meals[t])


nuts_kernel = NUTS(generate_traces, jit_compile=True, ignore_jit_warnings=True)
mcmc = MCMC(nuts_kernel, num_samples=100)
# we run MCMC to get posterior
mcmc.run([tensor(0), tensor(0)], [tensor(0), tensor(0)])
# after that, we plot the posterior
print(mcmc.get_samples()["transition_prob"])
"""
