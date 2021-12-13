from numbers import Number
from typing import Callable

from pyro.poutine.messenger import Messenger
from torch import Tensor


class SetValueMessenger(Messenger):

    def __init__(self, msg_type: str, authority):
        super().__init__()
        self.msg_type: str = msg_type
        self.authority = authority

    def _access(self, *args, **kwargs):
        if isinstance(self.authority, Callable):
            return self.authority(*args, **kwargs)
        elif any(isinstance(self.authority, t) for t in (Number, str, Tensor)):
            return self.authority
        # TODO: Accessible? At-able?
        raise NotImplementedError

    def _process_message(self, msg):
        if msg['type'] == self.msg_type:
            args = msg['args']
            kwargs = msg['kwargs']
            msg['value'] = self._access(*args, **kwargs)
        return None
