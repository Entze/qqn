from numbers import Number

import torch
from torch import Tensor

from qqn.library.action import action_collapse_type
from qqn.library.common import snd
from qqn.library.setvalue_messenger import SetValueMessenger


class MaxCollapseMessenger(SetValueMessenger):

    def __init__(self):
        super().__init__(action_collapse_type, None)

    def _access(self, estimations, *args, **kwargs):

        if isinstance(estimations, list) and len(estimations) > 0 and isinstance(estimations[0], tuple):
            estimations.sort(key=snd, reverse=True)
            if isinstance(estimations[0][1], Number):
                return estimations[0][1]
            elif isinstance(estimations[0][1], Tensor):
                estimations = list(map(snd, estimations))
                estimations = torch.stack(estimations)

        if isinstance(estimations, Tensor):
            return torch.max(estimations)

        raise NotImplementedError(
            f"Cannot select from ratings of type {type(estimations).__name__}, "
            f"you have to use a messenger that processes {str(action_collapse_type)}")

def max_collapse_agent():
    return MaxCollapseMessenger()