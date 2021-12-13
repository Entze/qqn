from collections import defaultdict
from numbers import Number
from typing import Callable

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.poutine.messenger import Messenger

from qqn.library.action import action_prior_eff
from qqn.library.common import const
from qqn.library.option import option_estimator_eff
from qqn.library.state import state_key_eff


class LearningAgentMessenger(Messenger):

    def __init__(self, optimization_steps=100, max_estimation_depth=None, alpha=1.0, gamma=0.0, max_cache_size=None):
        super().__init__()
        self.optimization_steps = optimization_steps
        self.max_estimation_depth = max_estimation_depth
        self.alpha = alpha
        self.gamma = gamma
        self.lru_cache_max_size = max_cache_size
        self.lru_cache_size = 0
        self.lru_cache_min_accessed = None
        self.lru_cache_accesses = defaultdict(int)
        self.lru_cache = dict()

    def policy(self, state):
        key = state_key_eff(state)
        if key in self.lru_cache:
            self.lru_cache_accesses[key] += 1
        else:
            optim, posterior = self._optimize(state)
            self.lru_cache[key]['optim'] = optim
            self.lru_cache[key]['value'] = posterior

        return self.lru_cache[key]['value']

    def _optimize(self, state):
        raise NotImplementedError

    def _model(self, state):
        prior = action_prior_eff(state)
        action_idx = pyro.sample("p_action_idx", dist.Categorical(logits=prior))
        with poutine.block():
            action_value = option_estimator_eff(state, action_idx, max_depth=self.max_estimation_depth)

        if isinstance(self.alpha, Number):
            alpha = self.alpha
        elif isinstance(self.alpha, Callable):
            alpha = self.alpha(state, action_idx)
        else:
            alpha = 1.

        if isinstance(self.gamma, Number):
            gamma = self.gamma
        elif isinstance(self.gamma, Callable):
            gamma = self.gamma(state, action_idx)
        else:
            gamma = 0.

        pyro.factor("p_action_factor", alpha * action_value - gamma)
        return action_idx

    def _guide(self, state):
        key = state_key_eff(state)
        posterior = pyro.param(f"p_preferences_{key}")
        action_idx = pyro.sample("p_action_idx", dist.Categorical(logits=posterior))
        return action_idx


class SamplingAgentMessenger(LearningAgentMessenger):
    pass


class SVIAgentMessenger(LearningAgentMessenger):
    pass
