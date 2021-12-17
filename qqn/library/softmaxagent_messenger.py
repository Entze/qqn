from numbers import Number

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import Distribution
from torch import Tensor, tensor

from qqn.library.action import action_select_type
from qqn.library.common import snd, fst
from qqn.library.setvalue_messenger import SetValueMessenger


class SoftmaxAgentMessenger(SetValueMessenger):

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
            return pyro.sample("action_selection", dist.Categorical(logits=ratings))

        raise NotImplementedError(
            f"Cannot select from ratings of type {type(ratings).__name__}, you have to use a messenger that processes {str(action_select_type)}")

    def process_message(self, msg):
        if msg['type'] == action_select_type:
            args = msg['args']
            estimations = args[0]
            value = msg['value']
            if isinstance(estimations, Tensor):
                value = pyro.sample("action_selection", dist.Categorical(logits=estimations))
            elif isinstance(estimations, list):
                if isinstance(estimations[0], Number):
                    value = pyro.sample("action_selection", dist.Categorical(logits=tensor(estimations)))
                elif isinstance(estimations[0], tuple):
                    es = sorted(estimations)
                    logits = tensor(list(map(snd, es))).float()
                    value = pyro.sample("action_selection", dist.Categorical(logits=logits))
            elif isinstance(estimations, Distribution):
                value = pyro.sample("action_selection", estimations)
            msg['value'] = value


def softmax_agent():
    return SoftmaxAgentMessenger()
