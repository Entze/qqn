import pyro
import pyro.distributions as dist
import pyro.infer
import torch
from pyro import poutine
from pyro.infer import Trace_ELBO, SVI
from pyro.optim import Adam
from torch import tensor
from torch.distributions.utils import logits_to_probs

from qqn.library.action import action_estimate_type, action_estimate_default, action_islegal_eff
from qqn.library.number_bin import NumberBin
from qqn.library.setvalue_messenger import SetValueMessenger
from qqn.library.state import state_key_eff

torch.autograd.set_detect_anomaly(True)


class EstimatorMessenger(SetValueMessenger):

    def __init__(self,
                 min_estimation_value=0, max_estimation_value=2, nr_of_bins=None,
                 optimization_steps=100,
                 max_estimator_depth=None):
        super().__init__(action_estimate_type, None)
        self.max_estimator_depth = max_estimator_depth
        self.binner = NumberBin(min_estimation_value, max_estimation_value, nr_of_bins)
        self.optimization_steps = optimization_steps
        self._action_value_prior_logits = torch.zeros(self.binner.nr_of_bins)
        self._action_value_prior = dist.Categorical(logits=self._action_value_prior_logits)

    def _access(self, action_idx, state, depth=0, max_depth=None, *args, **kwargs):
        if not action_islegal_eff(action_idx, state, *args, **kwargs):
            return torch.full_like(action_idx, float('-inf'), dtype=torch.float)
        return self._estimate(action_idx, state)

    def _estimate(self, action_idx, state):
        raise NotImplementedError

    def _calc(self, action_idx, state, depth=0, max_depth=None):
        return action_estimate_default(action_idx, state, depth, max_depth)

    def _model(self, action_idx, state):
        with poutine.block(hide_types=["sample"]):
            action_value = self._calc(action_idx, state, 0, self.max_estimator_depth)
        action_value_binned = self.binner.transform_to(action_value)
        pyro.sample("av_action_value_bin", self._action_value_prior, obs=action_value_binned)

        return action_value_binned

    def _guide(self, action_idx, state):
        state_key = state_key_eff(state)
        action_key = action_idx.item()
        posterior = pyro.param(f"av_preferences_{state_key}_{action_key}", torch.clone(self._action_value_prior_logits))
        action_value_binned = pyro.sample("av_action_value_bin", dist.Categorical(logits=posterior),
                                          infer={'is_auxiliary': True})
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


def sampling_estimating_agent(min_estimation_value=0, max_estimation_value=2, nr_of_bins=None,
                              optimization_steps=100,
                              max_estimator_depth=None):
    return SamplingEstimatingAgentMessenger(min_estimation_value, max_estimation_value, nr_of_bins, optimization_steps,
                                            max_estimator_depth)


def sampling_estimating_agent_gen(min_estimation_value=0, max_estimation_value=2, nr_of_bins=None,
                                  optimization_steps=100,
                                  max_estimator_depth=None):
    def gen():
        return sampling_estimating_agent(min_estimation_value, max_estimation_value, nr_of_bins, optimization_steps,
                                         max_estimator_depth)

    return gen


class SVIEstimatingAgentMessenger(EstimatorMessenger):

    def __init__(self,
                 min_estimation_value=0, max_estimation_value=2, nr_of_bins=10,
                 optimization_steps=100,
                 max_estimator_depth=None,
                 optim_generator=None,
                 optim_args=None,
                 optim_clip_args=None,
                 loss_generator=None,
                 loss_args=None,
                 loss_kwargs=None):
        super().__init__(min_estimation_value, max_estimation_value, nr_of_bins, optimization_steps,
                         max_estimator_depth)
        self.optim_generator = optim_generator
        self.optim_args = optim_args
        self.optim_clip_args = optim_clip_args
        self.loss_generator = loss_generator
        self.loss_args = loss_args
        self.loss_kwargs = loss_kwargs

        if self.optim_generator is None:
            self.optim_generator = Adam

        if self.optim_args is None:
            self.optim_args = {}

        if self.loss_generator is None:
            self.loss_generator = Trace_ELBO

        if self.loss_args is None:
            self.loss_args = []

        if self.loss_kwargs is None:
            self.loss_kwargs = {}

        self.precache = {}

    def _estimate(self, action_idx, state):
        state_key = state_key_eff(state)
        action_key = action_idx.item()

        if state_key not in self.precache:
            self.precache[state_key] = {}

        if action_key not in self.precache[state_key]:

            optim = self.optim_generator(self.optim_args, self.optim_clip_args)
            loss = self.loss_generator(*self.loss_args, **self.loss_kwargs)
            svi = SVI(model=self._model, guide=self._guide, optim=optim, loss=loss)

            self.precache[state_key][action_key] = torch.clone(self._action_value_prior_logits)
            for _ in range(self.optimization_steps):
                with poutine.block(hide=["av_action_value_bin"]):
                    svi.step(action_idx, state)
                    self.precache[state_key][action_key] = pyro.param(f"av_preferences_{state_key}_{action_key}")

        posterior = self.precache[state_key][action_key]
        probs = logits_to_probs(torch.clone(posterior))
        weighted_expected_action_value = probs * self.binner.bins_tensor
        action_value = weighted_expected_action_value.sum()
        return action_value


def svi_estimating_agent(min_estimation_value=0, max_estimation_value=2, nr_of_bins=10,
                         optimization_steps=100,
                         max_estimator_depth=None,
                         optim_generator=None,
                         optim_args=None,
                         optim_clip_args=None,
                         loss_generator=None,
                         loss_args=None,
                         loss_kwargs=None):
    return SVIEstimatingAgentMessenger(min_estimation_value, max_estimation_value, nr_of_bins, optimization_steps,
                                       max_estimator_depth, optim_generator, optim_args, optim_clip_args,
                                       loss_generator, loss_args, loss_kwargs)


def svi_estimating_agent_gen(min_estimation_value=0, max_estimation_value=2, nr_of_bins=10,
                             optimization_steps=100,
                             max_estimator_depth=None,
                             optim_generator=None,
                             optim_args=None,
                             optim_clip_args=None,
                             loss_generator=None,
                             loss_args=None,
                             loss_kwargs=None):
    def gen():
        return svi_estimating_agent(min_estimation_value, max_estimation_value, nr_of_bins, optimization_steps,
                                    max_estimator_depth, optim_generator, optim_args, optim_clip_args,
                                    loss_generator, loss_args, loss_kwargs)

    return gen
