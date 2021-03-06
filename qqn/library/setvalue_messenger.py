from numbers import Number

from torch import Tensor

from qqn.library.effect import Messenger


class SetValueMessenger(Messenger):

    def __init__(self, msg_type: str, authority):
        super().__init__()
        self.msg_type: str = msg_type
        self.authority = authority

    def _access(self, *args, **kwargs):
        if callable(self.authority):
            return self.authority(*args, **kwargs)
        elif any(isinstance(self.authority, t) for t in (Number, str, Tensor)):
            return self.authority
        elif hasattr(self.authority, "__getitem__"):
            authority = self.authority
            pointer = -1
            while pointer + 1 < len(args):
                pointer += 1
                authority = authority[args[pointer]]
            return authority
        raise NotImplementedError

    def process_message(self, msg):
        if not msg["done"] and msg["value"] is None and msg['type'] == self.msg_type:
            args = msg['args']
            kwargs = msg['kwargs']
            msg['value'] = self._access(*args, **kwargs)
            msg['done'] = True
