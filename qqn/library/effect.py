import functools
from functools import partial

# TODO: DOCUMENT PROPERLY

_QQN_STACK = []


def _context_wrap(context, fn, *args, **kwargs):
    with context:
        return fn(*args, **kwargs)


class _BoundPartial(partial):
    """
    Converts a (possibly) bound method into a partial function to
    support class methods as arguments to handlers.
    """
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return partial(self.func, instance)


class Messenger:
    """
    Context manager class that modifies behavior
    and adds side effects to functions.

    This is the base Messenger class.
    It implements the default behavior for all Pyro primitives,
    so that the joint distribution induced by a function fn is
     identical to the joint distribution induced by ``Messenger()(fn)``.

    Class of transformers for messages passed during inference.
    Most inference operations are implemented in subclasses of this.
    """

    def __init__(self):
        pass

    def __call__(self, fn):
        if not callable(fn):
            raise ValueError(
                "{} is not callable, did you mean to pass it as a keyword arg?".format(
                    fn
                )
            )
        wraps = _BoundPartial(partial(_context_wrap, self, fn))
        return wraps

    def __enter__(self):
        """
        :returns: self
        :rtype: qqn.effect.Messenger

        Installs this messenger at the bottom of the Pyro stack.

        Can be overloaded to add any additional per-call setup functionality,
        but the derived class must always push itself onto the stack, usually
        by calling super().__enter__().

        Derived versions cannot be overridden to take arguments
        and must always return self.
        """
        if not (self in _QQN_STACK):
            # if this handler is not already installed,
            # put it on the bottom of the stack.
            _QQN_STACK.append(self)

            # necessary to return self because the return value of __enter__
            # is bound to VAR in with EXPR as VAR.
            return self
        else:
            # note: currently we raise an error if trying to install a handler twice.
            # However, this isn't strictly necessary,
            # and blocks recursive handler execution patterns like
            # like calling self.__call__ inside of self.__call__
            # or with Handler(...) as p: with p: <BLOCK>
            # It's hard to imagine use cases for this pattern,
            # but it could in principle be enabled...
            raise ValueError("cannot install a Messenger instance twice")

    def __exit__(self, exc_type, exc_value, traceback):
        """
        :param exc_type: exception type, e.g. ValueError
        :param exc_value: exception instance?
        :param traceback: traceback for exception handling
        :returns: None
        :rtype: None

        Removes this messenger from the bottom of the Pyro stack.
        If an exception is raised, removes this messenger and everything below it.
        Always called after every execution of self.fn via self.__call__.

        Can be overloaded by derived classes to add any other per-call teardown functionality,
        but the stack must always be popped by the derived class,
        usually by calling super().__exit__(*args).

        Derived versions cannot be overridden to take other arguments,
        and must always return None or False.

        The arguments are the mandatory arguments used by a with statement.
        Users should never be specifying these.
        They are all None unless the body of the with statement raised an exception.
        """
        if exc_type is None:  # callee or enclosed block returned successfully
            # if the callee or enclosed block returned successfully,
            # this handler should be on the bottom of the stack.
            # If so, remove it from the stack.
            # if not, raise a ValueError because something really weird happened.
            if _QQN_STACK[-1] == self:
                _QQN_STACK.pop()
            else:
                # should never get here, but just in case...
                raise ValueError("This Messenger is not on the bottom of the stack")
        else:  # the wrapped function or block raised an exception
            # handler exception handling:
            # when the callee or enclosed block raises an exception,
            # find this handler's position in the stack,
            # then remove it and everything below it in the stack.
            if self in _QQN_STACK:
                loc = _QQN_STACK.index(self)
                for i in range(loc, len(_QQN_STACK)):
                    _QQN_STACK.pop()

    def _reset(self):
        pass

    def process_message(self, msg):
        """
        :param msg: current message at a trace site
        :returns: None

        Process the message by calling appropriate method of itself based
        on message type. The message is updated in place.
        """
        method = getattr(self, "_qqn_{}".format(msg["type"]), None)
        if method is not None:
            return method(msg)
        return None

    def postprocess_message(self, msg):
        method = getattr(self, "_qqn_post_{}".format(msg["type"]), None)
        if method is not None:
            return method(msg)
        return None

    @classmethod
    def register(cls, fn=None, type=None, post=None):
        """
        :param fn: function implementing operation
        :param str type: name of the operation
            (also passed to :func:`~qqn.effect.effectful`)
        :param bool post: if `True`, use this operation as postprocess

        Dynamically add operations to an effect.
        Useful for generating wrappers for libraries.

        Example::

            @SomeMessengerClass.register
            def some_function(msg)
                ...do_something...
                return msg

        """
        if fn is None:
            return lambda x: cls.register(x, type=type, post=post)

        if type is None:
            raise ValueError("An operation type name must be provided")

        setattr(cls, "_qqn_" + ("post_" if post else "") + type, staticmethod(fn))
        return fn

    @classmethod
    def unregister(cls, fn=None, type=None):
        """
        :param fn: function implementing operation
        :param str type: name of the operation
            (also passed to :func:`~qqn.effect.effectful`)

        Dynamically remove operations from an effect.
        Useful for removing wrappers from libraries.

        Example::

            SomeMessengerClass.unregister(some_function, "name")
        """
        if type is None:
            raise ValueError("An operation type name must be provided")

        try:
            delattr(cls, "_qqn_post_" + type)
        except AttributeError:
            pass

        try:
            delattr(cls, "_qqn_" + type)
        except AttributeError:
            pass

        return fn


def apply_stack(initial_msg):
    """
    Execute the effect stack at a single site according to the following scheme:

        1. For each ``Messenger`` in the stack from bottom to top,
           execute ``Messenger.process_message`` with the message;
           if the message field "stop" is True, stop;
           otherwise, continue
        2. Apply default behavior (``default_process_message``) to finish remaining site execution
        3. For each ``Messenger`` in the stack from top to bottom,
           execute ``postprocess_message`` to update the message and internal messenger state with the site results
        4. If the message field "continuation" is not ``None``, call it with the message

    :param dict initial_msg: the starting version of the trace site
    :returns: ``None``
    """
    stack = _QQN_STACK
    # TODO check at runtime if stack is valid

    # msg is used to pass information up and down the stack
    msg = initial_msg

    pointer = 0
    # go until time to stop?
    for frame in reversed(stack):

        pointer = pointer + 1

        frame.process_message(msg)

        if msg["stop"]:
            break

    default_process_message(msg)

    for frame in stack[-pointer:]:
        frame.postprocess_message(msg)

    cont = msg["continuation"]
    if cont is not None:
        cont(msg)

    return None


def am_i_wrapped():
    """
    Checks whether the current computation is wrapped in an messenger.
    :returns: bool
    """
    return len(_QQN_STACK) > 0


def effectful(fn=None, type=None):
    """
    :param fn: function or callable that performs an effectful computation
    :param str type: the type label of the operation, e.g. `"sample"`

    Wrapper for calling :func:`~qqn.effect.apply_stack` to apply any active effects.
    """
    if fn is None:
        return functools.partial(effectful, type=type)

    if getattr(fn, "_is_effectful", None):
        return fn

    assert type is not None, "must provide a type label for operation {}".format(fn)
    assert type != "message", "cannot use 'message' as keyword"

    @functools.wraps(fn)
    def _fn(*args, **kwargs):

        value = kwargs.pop("value", None)
        continuation = kwargs.pop("continuation", kwargs.pop("cont", None))

        if not am_i_wrapped():
            return fn(*args, **kwargs)
        else:
            msg = {
                "type": type,
                "fn": fn,
                "args": args,
                "kwargs": kwargs,
                "value": value,
                "done": False,
                "stop": False,
                "continuation": continuation,
            }
            # apply the stack and return its return value
            apply_stack(msg)
            return msg["value"]

    _fn._is_effectful = True
    return _fn


def default_process_message(msg):
    """
    Default method for processing messages in inference.

    :param msg: a message to be processed
    :returns: None
    """
    if msg["done"] or msg["value"] is not None:
        msg["done"] = True
        return msg

    msg["value"] = msg["fn"](*msg["args"], **msg["kwargs"])

    # after fn has been called, update msg to prevent it from being called again.
    msg["done"] = True
