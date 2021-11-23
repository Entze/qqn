from typing import List, Callable, Any, Optional

from effect import Delay, Effect, sync_performer, sync_perform, TypeDispatcher, ComposedDispatcher, base_dispatcher, \
    perform_delay_with_sleep


class Animate(object):

    def __init__(self, menu):
        self.menu = menu


class Interpolate(object):

    def __init__(self, start, stop, fun):
        self.start = start
        self.stop = stop
        self.fun = fun


class Menu:

    def __init__(self, brightness, size, selected):
        self.brightness = brightness,
        self.size = size
        self.selected = selected

    def update_brightness(self, new_brightness):
        self.brightness = new_brightness

    def __repr__(self):
        return f"b: {self.brightness}, si: {self.size}, sel: {self.selected}"

    def __str__(self):
        return repr(self)


@sync_performer
def interpolate(dispatcher, interpolate_intent: Interpolate):
    if interpolate_intent.start == interpolate_intent.stop:
        return None
    interpolate_intent.fun(interpolate_intent.start)
    sync_perform(dispatcher, Effect(Delay(1)))
    return Effect(Interpolate(interpolate_intent.start + 1, interpolate_intent.stop, interpolate_intent.fun))


@sync_performer
def animate(dispatcher, animate_intent: Animate):
    update_brightness_eff = Effect(Interpolate(0, 10, animate_intent.menu.update_brightness))

    return update_brightness_eff


def main():
    menu = Menu(0, 0, 0)

    animation_dispatcher = TypeDispatcher({
        Interpolate: interpolate,
        Animate: animate,
        Delay: perform_delay_with_sleep
    })

    dispatcher = ComposedDispatcher([
        animation_dispatcher,
        base_dispatcher
    ])

    print(menu)

    sync_perform(dispatcher, Effect(Animate(menu)))

    print(menu)


if __name__ == "__main__":
    main()
