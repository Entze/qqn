from numbers import Number

import torch
from torch import Tensor
from torch.distributions.utils import logits_to_probs, probs_to_logits

from qqn.library.action import action_rate_type
from qqn.library.common import snd
from qqn.library.setvalue_messenger import SetValueMessenger


class WeightedRateMessenger(SetValueMessenger):

    def __init__(self, alpha=1.0, gamma=0.0):
        super().__init__(action_rate_type, None)
        self.alpha = alpha
        self.gamma = gamma

    def _access(self, estimations, *args, **kwargs):
        assert len(args) > 0, "No estimations given."
        if isinstance(estimations, list) and len(estimations) > 0:
            if isinstance(estimations[0], Number):
                estimations = torch.as_tensor(estimations)
            elif isinstance(estimations[0], tuple):
                if isinstance(estimations[0][0], int):
                    estimations.sort()
                if isinstance(estimations[0][1], Number):
                    estimations = torch.tensor(list(map(snd, estimations)))
                elif isinstance(estimations[0][1], Tensor):
                    estimations = torch.stack(list(map(snd, estimations)))

        if isinstance(estimations, Tensor):
            estimations_unnormalized = estimations * self.__alpha(*args, **kwargs) - self.__gamma(*args, **kwargs)
            estimations = probs_to_logits(logits_to_probs(estimations_unnormalized))
            estimations[torch.isneginf(estimations_unnormalized)] = float('-inf')
            return estimations

        raise NotImplementedError(
            f"Cannot select from ratings of type {type(estimations).__name__}, "
            f"you have to use a messenger that processes {str(action_rate_type)}")

    def __alpha(self, *args, **kwargs):
        if callable(self.alpha):
            return self.alpha(*args, **kwargs)
        return self.alpha

    def __gamma(self, *args, **kwargs):
        if callable(self.gamma):
            return self.gamma(*args, **kwargs)
        return self.gamma
