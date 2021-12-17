from numbers import Number

import torch
from torch import Tensor

from qqn.library.action import action_select_type
from qqn.library.common import snd, fst
from qqn.library.setvalue_messenger import SetValueMessenger


class ArgmaxAgentMessenger(SetValueMessenger):

    def __init__(self):
        super().__init__(action_select_type, None)

    def _access(self, ratings, *args, **kwargs):

        if isinstance(ratings, list) and len(ratings) > 0:
            if isinstance(ratings[0], Number):
                ratings = torch.as_tensor(ratings)
            elif isinstance(ratings[0], tuple):
                ratings.sort(key=fst)
                estimations_l = list(map(snd, ratings))
                if isinstance(ratings[0][1], Number):
                    ratings = torch.as_tensor(estimations_l)
                elif isinstance(ratings[0][1], Tensor):
                    ratings = torch.stack(estimations_l)

        if isinstance(ratings, Tensor):
            return torch.argmax(ratings)

        raise NotImplementedError(
            f"Cannot select from ratings of type {type(ratings).__name__}, you have to use a messenger that processes {str(action_select_type)}")


def argmax_agent():
    return ArgmaxAgentMessenger()
