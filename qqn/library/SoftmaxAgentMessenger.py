from numbers import Number

import pyro
from pyro.distributions import Distribution
from pyro.poutine.messenger import Messenger
from torch import Tensor, tensor
import pyro.distributions as dist

from qqn.library.common import snd
from qqn.library.option import option_selector_type


class SoftmaxAgentMessenger(Messenger):

    def __init__(self, ):
        super().__init__()

    def _process_message(self, msg):
        if msg['type'] == option_selector_type:
            args = msg['args']
            estimations = args[1]
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


softmax_agent = SoftmaxAgentMessenger()
