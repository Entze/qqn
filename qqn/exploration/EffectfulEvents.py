from tkinter import Canvas, Tk, N, W, E, S
from typing import Dict, Tuple, Union, Optional, Callable, TypeVar, Any

from effect import Effect, sync_performer, TypeDispatcher, sync_perform, ComposedDispatcher

from qqn.exploration.GraphMachine import setdictkey, proceed, anim_orientation
from qqn.exploration.ScopedTypeDispatcher import ScopedTypeDispatcher

example_state = {
    "id": 1,
    "observation": []
}

example_events = [
    {
        "name": "pressed key",
        "pressed_key": "i",
        "update": lambda s, _: s
    },
    {
        "name": "click",
        "pointer_x": 100,
        "pointer_y": 100,
        "update": lambda s, e: setdictkey(setdictkey(s, "pointer_x", e["pointer_x"]), "pointer_y",
                                          e["pointer_y"]),
    },
    {
        "name": "pressed key",
        "pressed_key": "i",
        "update": lambda s, _: s
    },
    {
        "name": "click_cell",
        "update": lambda s, e: setdictkey(s, "hit_node", e["hit_node"]),
        "hit_node": 1},
    {
        "name": "drag",
        "update": lambda s, e: setdictkey(setdictkey(s, "position_x", e["pointer_x"]), "position_y",
                                          e["pointer_y"]),
        "pointer_x": 100,
        "pointer_y": 100,
    },
    {
        "name": "drag",
        "update": lambda s, e: setdictkey(setdictkey(s, "position_x", e["pointer_x"]), "position_y",
                                          e["pointer_y"]),
        "pointer_x": 300,
        "pointer_y": 100,
    },
    {
        "name": "release",
        "update": lambda s, e: s
    }
]

e = example_events.copy()


def simple_event_chain(state=None, events=None):
    if events is None:
        events = example_events
    if state is None:
        state = example_state
    trace = []
    while len(events) > 0:
        if len(state["observation"]) == 0:
            state["observation"].append(events[0])
            events = events[1:]
        trace = trace[:-1] + proceed(state, anim_orientation, 1)
        assert trace
        state = trace[-1]
    return trace


class DrawIntent(object):
    def __init__(self, state, observation_src):
        self.state = state
        self.observation_src = observation_src


class DisplayIntent(object):
    def __init__(self, state):
        self.state = state


def draw_effect(state, observation_src):
    return Effect(DrawIntent(state, observation_src))


def display_effect(state):
    return Effect(DisplayIntent(state))


def simple_observation_src():
    global example_events
    if not example_events:
        return None
    next_event = example_events.pop(0)
    return next_event


@sync_performer
def print_state(dispatcher, display_intent: DisplayIntent):
    state = display_intent.state
    if state["id"] == 1:
        print("View")
    elif state["id"] == 2:
        if 'position_x' in state:
            print(f"Move({state['hit_node']}) to {state['position_x']},{state['position_y']}")
        else:
            print(f"Move({state['hit_node']})")
    elif state["id"] == 3:
        if "pointer_x" in state:
            print(f"Created Node at ({state['pointer_x']},{state['pointer_y']})")
        else:
            print("Insert Node State")
    else:
        print("Unknown State")


@sync_performer
def simple_draw(dispatcher, draw_intent: DrawIntent):
    next_event = draw_intent.observation_src()
    state = draw_intent.state
    trace = []
    while next_event is not None:
        sync_perform(dispatcher, display_effect(state))
        if "observation" not in state:
            state["observation"] = []
        if not state["observation"]:
            state["observation"].append(next_event)
            next_event = draw_intent.observation_src()
        trace = trace[:-1] + proceed(state, anim_orientation, 3)
        assert trace
        state = trace[-1]
    sync_perform(dispatcher, display_effect(state))
    return trace


class CanvasObject:
    def __init__(self):
        pass

    def in_convex_hull(self, p: Union[Tuple[int, int], int], q: Optional[int] = None) -> bool:
        if isinstance(p, tuple):
            return self._in_convex_hull_point(p[0], p[1])
        elif isinstance(p, int) and isinstance(q, int):
            return self._in_convex_hull_point(p, q)
        raise Exception(
            f"Either the first argument should be a tuple or an int with the second coordinate as int. However p: {type(p)} and q: {type(q)}")

    def _in_convex_hull_point(self, x: int, y: int) -> bool:
        return False

    def translate(self, p: Union[Tuple[int, int], int], q: Optional[int] = None) -> None:
        if isinstance(p, tuple):
            return self._translate(p[0], p[1])
        elif isinstance(p, int) and isinstance(q, int):
            return self._translate(p, q)
        raise Exception(
            f"Either the first argument should be a tuple or an int with the second coordinate as int. However p: {type(p)} and q: {type(q)}")

    def _translate(self, x: int, y: int) -> None:
        return None

    def draw_on_canvas(self, canvas: Canvas) -> Optional[int]:
        return None


class Rectangle(CanvasObject):
    def __init__(self, left_top_x: int, left_top_y: int, width: int, height: Optional[int] = None):
        super(Rectangle, self).__init__()
        if height is None:
            height = width
        self.left_top_x = left_top_x
        self.left_top_y = left_top_y
        self.width = width
        self.height = height
        self._fhalfwidth = width // 2
        self._fhalfheight = height // 2
        self._lhalfwidth = self.width - self._fhalfwidth
        self._lhalfheight = self.height - self._fhalfheight
        self.right_bottom_x = left_top_x + width
        self.right_bottom_y = left_top_y + height

    def _in_convex_hull_point(self, x: int, y: int) -> bool:
        return self.left_top_x <= x <= self.right_bottom_x and \
               self.left_top_y <= y <= self.right_bottom_y

    def draw_on_canvas(self, canvas: Canvas) -> Optional[int]:
        return canvas.create_rectangle(self.left_top_x, self.left_top_y, self.right_bottom_x, self.right_bottom_y,
                                       fill="blue")

    def _translate(self, x: int, y: int) -> None:
        self.left_top_x = x - self._fhalfwidth
        self.left_top_y = y - self._fhalfheight
        self.right_bottom_x = x + self._lhalfwidth
        self.right_bottom_y = y + self._lhalfheight


class Sketchpad(Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.drawn_objects: Dict[int, CanvasObject] = {}
        self.active_objects: Dict[int, bool] = {}

    def draw_rectangle(self, center_x: int, center_y: int, width: int, height: Optional[int] = None):
        if height is None:
            height = width
        fhalfwidth = width // 2
        fhalfheight = height // 2
        left_top_x = center_x - fhalfwidth
        left_top_y = center_y - fhalfheight
        rect = Rectangle(left_top_x, left_top_y, width, height)
        i = rect.draw_on_canvas(self)
        if i is not None:
            self.drawn_objects[i] = rect
            self.active_objects[i] = True

    def object_at_point(self, p: Union[Tuple[int, int]], q: Optional[int] = None) -> Optional[Tuple[int, CanvasObject]]:
        if isinstance(p, tuple):
            return self._object_at_point(p[0], p[1])
        elif isinstance(p, int) and isinstance(q, int):
            return self._object_at_point(p, q)
        raise Exception(
            f"Either the first argument should be a tuple or an int with the second coordinate as int. However p: {type(p)} and q: {type(q)}")

    def _object_at_point(self, x: int, y: int) -> Optional[Tuple[int, CanvasObject]]:
        for i, a in self.active_objects.items():
            if a and self.drawn_objects[i].in_convex_hull(x, y):
                return i, self.drawn_objects[i]
        return None

    def redraw_objects(self):
        self.delete("all")
        new_drawn_objects = {}
        new_active_objects = {}

        for i, a in self.active_objects.items():
            if a:
                ni = self.drawn_objects[i].draw_on_canvas(self)
                new_drawn_objects[ni] = self.drawn_objects[i]
            else:
                pass


class Displayer:
    Displayable = TypeVar('Displayable')

    def __init__(self, obj: Displayable, display_func: Callable[[Displayable], Any],
                 is_alive_func: Optional[Callable[[Displayable], bool]] = None):
        self.obj = obj
        self.display_func = display_func
        self.is_alive_func = is_alive_func
        if self.is_alive_func is None:
            self.is_alive_func = lambda _: True

    def display(self):
        return self.display_func(self.obj)

    def is_alive(self):
        return self.is_alive_func(self.obj)


def extract_displayer_and_canvas(dispatcher) -> Optional[Tuple[Displayer, Sketchpad]]:
    displayer: Optional[Tuple[Displayer, Sketchpad]] = None
    dispatchers = []
    if isinstance(dispatcher, ScopedTypeDispatcher) and isinstance(dispatcher.scope, dict):
        dispatchers.append(dispatcher)
    elif isinstance(dispatcher, ComposedDispatcher):
        dispatchers += [d for d in dispatcher.dispatchers if
                        isinstance(d, ScopedTypeDispatcher) and isinstance(d.scope, dict)]
    for d in dispatchers:
        if "displayer" in d.scope and "canvas" in d.scope:
            displayer = d.scope["displayer"], d.scope["canvas"]
            break
    return displayer


@sync_performer
def display_canvas(dispatcher, display_intent: DisplayIntent):
    displayer, canvas = extract_displayer_and_canvas(dispatcher)
    if displayer is None:
        raise Exception("Dispatcher is not able to display canvas")
    assert isinstance(displayer, Displayer)
    assert isinstance(canvas, Sketchpad)

    state: Dict = display_intent.state

    if "id" in state and state["id"] == 3 and "pointer_x" in state:
        canvas.draw_rectangle(state["pointer_x"], state["pointer_y"], 50)

    if "id" in state and state["id"] == 2 and "hit_node" in state and "position_x" in state:
        hit_node = state["hit_node"]
        canvas.drawn_objects[hit_node].translate(state["position_x"], state["position_y"])
        canvas.redraw_object(hit_node)

    displayer.display()
    return displayer.is_alive()


@sync_performer
def scoped_draw(dispatcher, draw_intent: DrawIntent):
    next_event = draw_intent.observation_src()
    state = draw_intent.state
    trace = []
    alive_l: bool = True

    while alive_l:
        alive_l = sync_perform(dispatcher, display_effect(state))
        if next_event is None:
            next_event = draw_intent.observation_src()
        if "observation" not in state:
            state["observation"] = []
        if not state["observation"] and next_event is not None:
            state["observation"].append(next_event)
            next_event = None
        trace = trace[:-1] + proceed(state, anim_orientation, 1)
        assert trace
        state = trace[-1]


def loop(tk: Tk):
    return tk.update()


alive: bool = True


def check_alive(tk: Tk):
    return alive


def on_delete():
    global alive
    alive = False


def main():
    trace = simple_event_chain()
    print("Without effects:", trace)

    global example_events
    global e
    example_events = e.copy()

    sync_perform(
        TypeDispatcher({
            DrawIntent: simple_draw,
            DisplayIntent: print_state
        }),
        draw_effect({"id": 1}, simple_observation_src))

    root = Tk()
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    root.protocol("WM_DELETE_WINDOW", on_delete)

    sketch = Sketchpad(root)
    sketch.grid(column=0, row=0, sticky=(N, W, E, S))

    example_events = e.copy()
    sync_perform(
        ScopedTypeDispatcher(mapping={
            DrawIntent: scoped_draw,
            DisplayIntent: display_canvas
        },
            scope={"displayer": Displayer(root, loop, check_alive), "canvas": sketch}),
        draw_effect({"id": 1}, simple_observation_src)
    )


if __name__ == "__main__":
    main()
