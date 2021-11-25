import attr


@attr.s
class ScopedTypeDispatcher(object):
    """
    An Effect dispatcher which looks up the performer to use by type.

    :param mapping: mapping of intent type to performer
    :param scope: scope of dispatcher
    """

    mapping = attr.ib()
    scope = attr.ib()

    def __call__(self, intent):
        return self.mapping.get(type(intent))
