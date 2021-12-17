from functools import partial

from qqn.library.effect import Messenger


def _block_fn(expose_types, hide_types, hide_all, msg):
    # handle observes
    msg_type = msg["type"]

    is_not_exposed = msg_type not in expose_types

    # decision rule for hiding:
    return (msg_type in hide_types) or (is_not_exposed and hide_all)


def _make_default_hide_fn(hide_all, expose_all, hide_types, expose_types):
    # first, some sanity checks:
    # hide_all and expose_all intersect?
    assert (hide_all is False and expose_all is False) or (
            hide_all != expose_all
    ), "cannot hide and expose an effect"


    # hide_types and expose_types intersect?
    if hide_types is None:
        hide_types = []
    else:
        hide_all = False

    if expose_types is None:
        expose_types = []
    else:
        hide_all = True

    assert set(hide_types).isdisjoint(
        set(expose_types)
    ), "cannot hide and expose an effect type"

    return partial(_block_fn, expose_types, hide_types, hide_all)


class BlockMessenger(Messenger):

    def __init__(self,
                 hide_fn=None,
                 expose_fn=None,
                 hide_all=True,
                 expose_all=False,
                 hide_types=None,
                 expose_types=None):
        super().__init__()
        if not (hide_fn is None or expose_fn is None):
            raise ValueError("Only specify one of hide_fn or expose_fn")
        if hide_fn is not None:
            self.hide_fn = hide_fn
        elif expose_fn is not None:
            self.hide_fn = lambda msg: not expose_fn(msg)
        else:
            self.hide_fn = _make_default_hide_fn(
                hide_all, expose_all, hide_types, expose_types
            )

    def process_message(self, msg):
        msg['stop'] = bool(self.hide_fn(msg))


def block(*args, **kwargs):
    return BlockMessenger(*args, **kwargs)
