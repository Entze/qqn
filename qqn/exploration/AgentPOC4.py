import random
import pyro
import pyro.infer
import pyro.distributions as dist
import torch
from pyro.infer import Trace_ELBO, SVI
from torch import tensor
from tqdm import trange

states = [1, 2, 3]
actions = [1, 2]
alpha = 1


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
        self.action_num = 0
        self.cache = {}
        self.optimizer = pyro.optim.Adam({"lr": 0.025})

    def _next_action_num(self):
        self.action_num += 1
        return self.action_num

    def act_dist(self, state):
        global_run = self._next_action_num()
        if state not in self.cache:
            svi = SVI(model=self.act, guide=self.act_guide, optim=self.optimizer, loss=Trace_ELBO())
            for i in range(100):
                svi.step(state, global_run, i)
            self.cache[state] = svi.marginal(sites=[f"{self.name}_action"])
        return self.cache[state]

    def act(self, state, global_run=0, run=0):
        action_dist = dist.Categorical(tensor([.5, .5]))
        action_t = pyro.sample(f"{self.name}_action", action_dist)
        action = actions[action_t.item()]
        eu = self.expected_utility(state, action)
        print(eu)
        pyro.factor(f"{self.name}_obs", tensor(eu) * 100)
        return action

    def act_guide(self, state, global_run=0, run=0):
        action_preference = pyro.param("preference", torch.zeros(2))
        action_dist = dist.Categorical(logits=action_preference)
        action_t = pyro.sample(f"{self.name}_action", action_dist)
        action = actions[action_t.item()]
        # eu = self.expected_utility(state, action)
        # print(eu)
        # pyro.factor(f"{self.name}_obs", tensor(eu) * 100)
        return action

    def expected_utility(self, state, action):
        u = reward(state, action)
        if is_final_state(state, action):
            pyro.deterministic("expected_util", tensor(u))
            return float(u)
        next_state = transition(state, action)
        expected_util_trace = pyro.infer.Importance(model=self._expected_utility_model, num_samples=10).run(next_state)
        expected_util_dist = pyro.infer.EmpiricalMarginal(expected_util_trace, sites=["expected_util"])
        eu = u + expected_util_dist.sample_n(10).float().mean().item()
        eu_p = pyro.deterministic("expected_util", tensor(eu))
        return eu

    def _expected_utility_model(self, next_state):
        next_action = pyro.sample(f"{self.name}_next_action_{self._next_action_num()}", self.act_dist(next_state))
        return self.expected_utility(next_state, next_action)


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
