import math
from collections import defaultdict
from numbers import Number
from typing import Callable

import pyro
import pyro.infer
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.poutine.messenger import Messenger
from torch import tensor
from torch.distributions.utils import logits_to_probs

from qqn.library.action import action_prior_eff, all_actions_eff
from qqn.library.common import const
from qqn.library.number_bin import NumberBin
from qqn.library.action import action_estimate_eff
from qqn.library.policy import policy_type, policy_posterior_type, policy_distribution_type
from qqn.library.state import state_key_eff


class LearningAgentMessenger(Messenger):

    def __init__(self,
                 policy_optimization_steps=100, action_value_optimization_steps=100,
                 max_estimation_depth=None,
                 alpha=1.0, gamma=0.0,
                 min_state_value=0.,
                 max_state_value=1.,
                 number_of_bins=100,
                 max_size_policy_cache=None, max_size_action_value_cache=None):
        super().__init__()
        self.policy_optimization_steps = policy_optimization_steps
        self.action_value_optimization_steps = action_value_optimization_steps
        self.max_estimation_depth = max_estimation_depth
        self.alpha = alpha
        self.gamma = gamma
        self.binner = NumberBin(min(min_state_value, min_state_value * 2),
                                max(max_state_value, max_state_value * 2),
                                number_of_bins)
        self.policy_cache = dict()
        self.action_value_cache = dict()
        self.all_actions = all_actions_eff()

    def action_value_posterior(self, state, action_idx):
        state_key = state_key_eff(state)
        action_key = action_idx.item()
        if state_key in self.action_value_cache and action_key in self.action_value_cache[state_key]:
            pass
        elif state_key not in self.action_value_cache:
            self.action_value_cache[state_key] = {}
        if action_key not in self.action_value_cache[state_key]:
            self.action_value_cache[state_key][action_key] = {}
            posterior = pyro.param(f"av_preferences_{state_key}_{action_key}", torch.zeros(self.binner.nr_of_bins))
            optim, marginal, posterior = self._optimize_action_value(state, action_idx)
            self.action_value_cache[state_key][action_key]['optim'] = optim
            self.action_value_cache[state_key][action_key]['marginal'] = marginal
            self.action_value_cache[state_key][action_key]['posterior'] = posterior
            self.action_value_cache[state_key][action_key]['distribution'] = dist.OneHotCategorical(logits=posterior)
        return self.action_value_cache[state_key][action_key]['posterior']

    def action_value_distribution(self, state, action_idx):
        state_key = state_key_eff(state)
        action_key = action_idx.item()
        self.action_value_posterior(state, action_idx)
        return self.action_value_cache[state_key][action_key]['distribution']

    def action_value(self, state, action_idx):
        probs = logits_to_probs(self.action_value_posterior(state, action_idx))
        action_value = (probs * self.binner.bins_tensor).sum()
        return action_value

    def action_value_bin(self, state, action_idx):
        return self.binner.transform_to(self.action_value(state, action_idx))

    def policy_posterior(self, state):
        key = state_key_eff(state)
        if key in self.policy_cache:
            pass
        else:
            self.policy_cache[key] = {}
            posterior = pyro.param(f"p_preferences_{key}", action_prior_eff(state))
            optim, marginal, posterior = self._optimize_policy(state)
            self.policy_cache[key]['optim'] = optim
            self.policy_cache[key]['marginal'] = marginal
            self.policy_cache[key]['posterior'] = posterior
            self.policy_cache[key]['distribution'] = dist.Categorical(logits=posterior)

        return self.policy_cache[key]['posterior']

    def policy_distribution(self, state):
        key = state_key_eff(state)
        self.policy_posterior(state)
        return self.policy_cache[key]['distribution']

    def policy(self, state):
        return self.policy_distribution(state).sample()

    def _process_message(self, msg):
        args = msg['args']
        if msg['type'] == policy_type:
            state = args[0]
            msg['value'] = self.policy(state)
        elif msg['type'] == policy_posterior_type:
            state = args[0]
            msg['value'] = self.policy_posterior(state)
        elif msg['type'] == policy_distribution_type:
            state = args[0]
            msg['value'] = self.policy_distribution(state)

    def _optimize_policy(self, state):
        raise NotImplementedError

    def _policy_model(self, state):
        prior = action_prior_eff(state)
        action_idx = pyro.sample("p_action_idx", dist.Categorical(logits=prior))
        action_value = self.action_value(state, action_idx)

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

    def _policy_guide(self, state):
        key = state_key_eff(state)
        posterior = pyro.param(f"p_preferences_{key}")
        action_idx = pyro.sample("p_action_idx", dist.Categorical(logits=posterior))
        return action_idx

    def _optimize_action_value(self, state, action_idx):
        raise NotImplementedError

    def _action_value_model(self, state, action_idx):
        with poutine.block(hide=["p_action_idx"]):
            action_value = action_estimate_eff(state, action_idx, max_depth=self.max_estimation_depth)
        action_value_binned = self.binner.transform_to(action_value)
        pyro.sample(f"av_action_value_bin", dist.Categorical(logits=torch.zeros(self.binner.nr_of_bins)),
                    obs=action_value_binned)
        return action_value_binned

    def _action_value_guide(self, state, action_idx):
        state_key = state_key_eff(state)
        action_key = action_idx.item()
        posterior = pyro.param(f"av_preferences_{state_key}_{action_key}")
        action_value_binned = pyro.sample(f"av_action_value_bin", dist.Categorical(logits=posterior))
        return action_value_binned


class SamplingAgentMessenger(LearningAgentMessenger):

    def _optimize_policy(self, state):
        importance = pyro.infer.Importance(model=self._policy_model, num_samples=self.policy_optimization_steps)
        importance.run(state)
        marginal = pyro.infer.EmpiricalMarginal(importance)
        logits = torch.zeros(len(self.all_actions))
        for a in self.all_actions:
            logits[a] = marginal.log_prob(a)
        return None, marginal, logits

    def _optimize_action_value(self, state, action_idx):
        importance = pyro.infer.Importance(model=self._action_value_model,
                                           num_samples=self.action_value_optimization_steps)
        importance.run(state, action_idx)
        marginal = pyro.infer.EmpiricalMarginal(importance)
        logits = torch.zeros(self.binner.nr_of_bins)
        for b in range(self.binner.nr_of_bins):
            logits[b] = marginal.log_prob(tensor(b))
        return None, marginal, logits


class SVIAgentMessenger(LearningAgentMessenger):
    pass
