from collections import defaultdict

from pyro import markov
from pyroapi import pyro
import pyro.infer
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
from pyro.infer import Trace_ELBO, SVI
from torch import tensor
from tqdm import trange

states = [1, 2, 3]
actions = [1, 2]
alpha = 1

torch.manual_seed(0)
pyro.set_rng_seed(0)


def transition(s, a):
    return {
        (0, 1): 1,
        (1, 1): 2,
        (1, 2): 3,
        (2, 1): 3,
    }.get((s, a), None)


weights = {
    (0, 1): 1.,
    (1, 1): .25,
    (1, 2): .75,
    (2, 1): 1.
}


def reward(s, a):
    return {
        (0, 1): 0,
        (1, 1): 50,
        (1, 2): 1,
        (2, 1): 3
    }.get((s, a), 0)


def is_final_state(s, a):
    return transition(s, a) is None


def state_to_action(s):
    return [act for act in actions if transition(s, act) is not None]


class Agent:

    def __init__(self, name="Agent"):
        self.name = name
        self.act_dist_cache = {}

    def act_dist(self, state):
        if state not in self.act_dist_cache:
            self.act_dist_cache[state] = dict(
                done=False,
                marginal=dist.Categorical(logits=torch.zeros(2)),
                sample_model=0,
                sample_guide=0,
                iteration=0
            )
            print("\t\tOptimizing state", state)
            optimizer = pyro.optim.Adam({"lr": 0.025})
            svi = SVI(model=self.act, guide=self.act_guide, optim=optimizer, loss=Trace_ELBO())
            self.act_dist_cache[state]['iteration'] += 1
            for i in range(100):
                svi.step(state, self.act_dist_cache[state]['iteration'], i)
                if i % 20 == 19:
                    self.act_dist_cache[state]['marginal'] = svi.marginal()
            self.act_dist_cache[state]['done'] = True
            print("Done\toptimizing state", state)
        return self.act_dist_cache[state]['marginal']

    def act(self, state, attempt=None, step=None):
        print("\t\tModel step", step, "state", state, "attempt", attempt)
        action_dist = dist.Categorical(tensor([.5, .5]))
        sample_model = self.act_dist_cache[state]['sample_model']
        action_t = pyro.sample(f"{self.name}_action_{state}_{attempt}_{step}_{sample_model}", action_dist)
        self.act_dist_cache[state]['sample_model'] += 1
        action = actions[action_t.item()]
        with poutine.block():
            eu = self.expected_utility(state, action)
        pyro.factor(f"{self.name}_obs", tensor(eu) * 100)
        print("Done\tModel step", step, "state", state, "attempt", attempt, "eu", eu)
        return action_t

    def act_guide(self, state, attempt=None, step=None):
        print("\t\tGuide step", step, "state", state, "attempt", attempt)
        action_preference = pyro.param("preference", torch.zeros(2))
        action_dist = dist.Categorical(logits=action_preference)
        sample_guide = self.act_dist_cache[state]['sample_guide']
        action_t = pyro.sample(f"{self.name}_action_{state}_{attempt}_{step}_{sample_guide}", action_dist)
        self.act_dist_cache[state]['sample_guide'] += 1
        action = actions[action_t.item()]
        # eu = self.expected_utility(state, action)
        # print(eu)
        # pyro.factor(f"{self.name}_obs", tensor(eu) * 100)
        print("Done\tGuide step", step, "state", state, "attempt", attempt)
        return action_t

    def expected_utility(self, state, action):
        u = reward(state, action)
        if is_final_state(state, action):
            pyro.deterministic("expected_util", tensor(u))
            return float(u)
        next_state = transition(state, action)
        expected_util_trace = pyro.infer.Importance(model=self._expected_utility_model, num_samples=100).run(next_state)
        expected_util_dist = pyro.infer.EmpiricalMarginal(expected_util_trace, sites=["expected_util"])
        eu = u + expected_util_dist.sample_n(10).float().mean().item()
        eu_p = pyro.deterministic("expected_util", tensor(eu))
        return eu

    def _expected_utility_model(self, next_state):
        act_dist = self.act_dist(next_state)  # TODO: Case where optimization has started but is not finished yet
        next_act = pyro.sample(f"{self.name}_next_action", act_dist)
        return self.expected_utility(next_state, next_act)


agent = Agent()

agent_dist = agent.act_dist(0)
next_action_t = agent_dist.sample()
next_action = actions[next_action_t.item()]
print(next_action, transition(0, next_action))

stats = {
    1: 0,
    2: 0
}

for _ in trange(50000):
    a_t = agent_dist.sample()
    a = actions[a_t.item()]
    stats[a] += 1

print(stats)
