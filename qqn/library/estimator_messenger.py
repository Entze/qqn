import pyro
import pyro.distributions as dist
import pyro.infer
import torch
from pyro import poutine
from torch import tensor
from torch.distributions.utils import logits_to_probs

from qqn.library.action import action_estimate_type, action_estimate_default
from qqn.library.number_bin import NumberBin
from qqn.library.setvalue_messenger import SetValueMessenger
from qqn.library.state import state_key_eff


class EstimatorMessenger(SetValueMessenger):

    def __init__(self,
                 min_estimation_value=0, max_estimation_value=2, nr_of_bins=10,
                 optimization_steps=100,
                 max_estimator_depth=None):
        super().__init__(action_estimate_type, None)
        self.max_estimator_depth = max_estimator_depth
        self.binner = NumberBin(min_estimation_value, max_estimation_value, nr_of_bins)
        self.optimization_steps = optimization_steps
        self.__action_value_prior_logits = torch.zeros(self.binner.nr_of_bins)
        self.__action_value_prior = dist.Categorical(logits=self.__action_value_prior_logits)

    def _access(self, action_idx, state, depth=0, max_depth=None):
        return self._estimate(action_idx, state)

    def _estimate(self, action_idx, state):
        raise NotImplementedError

    def _calc(self, action_idx, state, depth=0, max_depth=None):
        return action_estimate_default(action_idx, state, depth, max_depth)

    def _model(self, action_idx, state):
        with poutine.block(hide_types=["sample"]):
            action_value = self._calc(action_idx, state, 0, self.max_estimator_depth)
        action_value_binned = self.binner.transform_to(action_value)
        pyro.sample("av_action_value_bin", self.__action_value_prior, obs=action_value_binned)
        return action_value_binned

    def _guide(self, action_idx, state):
        state_key = state_key_eff(state)
        action_key = action_idx.item()
        posterior = pyro.param(f"av_preferences_{state_key}_{action_key}", self.__action_value_prior_logits)
        action_value_binned = pyro.sample("av_action_value_bin", dist.Categorical(logits=posterior))
        return action_value_binned


class SamplingEstimatingAgentMessenger(EstimatorMessenger):

    def _estimate(self, action_idx, state):
        importance = pyro.infer.Importance(model=self._model, num_samples=self.optimization_steps)
        importance.run(action_idx, state)
        marginal = pyro.infer.EmpiricalMarginal(importance)
        logits = torch.zeros(self.binner.nr_of_bins)
        for b in range(self.binner.nr_of_bins):
            logits[b] = marginal.log_prob(tensor(b))
        probs = logits_to_probs(logits)
        weighted_expected_action_value = probs * self.binner.bins_tensor
        action_value = weighted_expected_action_value.sum()
        return action_value
