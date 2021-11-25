import functools
from time import sleep
from tkinter import Tk, Canvas, N, W, E, S
from typing import Dict, Union, Tuple, Optional

from effect import sync_perform, Effect, sync_performer, TypeDispatcher

from qqn.ScopedTypeDispatcher import ScopedTypeDispatcher


class ViewIntent(object):
    def __init__(self, state):
        self.state = state


class MoveIntent(object):
    def __init__(self, node_id, cursor_x, cursor_y):
        self.node_id = node_id
        self.cursor_x = cursor_x
        self.cursor_y = cursor_y


class OutsideOutput:
    def __init__(self, dispatcher):
        self.dispatcher = dispatcher
        self.alive = True

    def step(self):
        pass

    def perform(self, effect):
        return sync_perform(self.dispatcher, effect)


class CmdOutsideOutput(OutsideOutput):

    @staticmethod
    @sync_performer
    def perform_view(dispatcher, view_intent: ViewIntent):
        print("view-state", view_intent.state)

    @staticmethod
    @sync_performer
    def perform_move(dispatcher, move_intent: MoveIntent):
        print("move-state", move_intent.node_id, move_intent.cursor_x, move_intent.cursor_y)


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
        self.draw_rectangle(10, 20, 50, 50)

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

    def redraw_object(self, i):
        if self.active_objects[i]:
            self.delete(i)
            obj = self.drawn_objects[i]
            new_i = obj.draw_on_canvas(self)
            self.active_objects[new_i] = True
            self.drawn_objects[new_i] = obj
            del self.active_objects[i]
            del self.drawn_objects[i]


class TkInterOutsideOutput(OutsideOutput):

    def __init__(self, dispatcher):
        super(TkInterOutsideOutput, self).__init__(dispatcher)
        assert isinstance(dispatcher, ScopedTypeDispatcher)
        dispatcher.scope = self
        self.tk_root: Tk = Tk()
        self.tk_root.columnconfigure(0, weight=1)
        self.tk_root.rowconfigure(0, weight=1)
        self.alive: bool = True

        self.tk_root.protocol("WM_DELETE_WINDOW", self.__on_delete)

        self.sketch: Sketchpad = Sketchpad(self.tk_root)
        self.sketch.grid(column=0, row=0, sticky=(N, W, E, S))

    def __on_delete(self):
        self.alive = False

    def step(self):
        self.tk_root.update()

    @staticmethod
    @sync_performer
    def perform_view(dispatcher, view_intent: ViewIntent):
        # print("In View-Mode")
        pass

    @staticmethod
    @sync_performer
    def perform_move(dispatcher: ScopedTypeDispatcher, move_intent: MoveIntent):
        sketch: Sketchpad = dispatcher.scope.sketch
        node_id = move_intent.node_id
        sketch.drawn_objects[node_id].translate(move_intent.cursor_x, move_intent.cursor_y)
        sketch.redraw_object(node_id)


class OutsideInput:

    def poll(self):
        return None


class CmdOutsideInput(OutsideInput):
    def __init__(self):
        pass

    def poll(self):
        cmd = input("> ")
        return {"name": "Key", "symb": cmd}


class TkInterOutsideIntput(OutsideInput):

    def __init__(self, sketch):
        self.sketch: Sketchpad = sketch
        self.queue = []
        self.sketch.bind("<Button-1>", self.__insert_to_queue_button)
        self.sketch.focus_set()
        self.sketch.bind("<Key>", self.__insert_to_queue_key)

    def __insert_to_queue_button(self, event):
        self.queue.append({"name": f"Button-{event.num}", "cursorX": event.x, "cursorY": event.y})

    def __insert_to_queue_key(self, event):
        self.queue.append({"name": "Key", "symb": event.keysym})

    def poll(self):
        if self.queue:
            return self.queue.pop(0)
        return None


view_to_move = {
    "name": "view to move",
    "state_check": [lambda s: s["intent"] == "view", lambda s: "observation" in s, lambda s: bool(s["observation"]),
                    lambda s: "node" in s["observation"]],
    "state_update": [
        lambda s: {"intent": "move", "node": s["observation"]["node"], "cursorX": s["observation"]["cursorX"],
                   "cursorY": s["observation"]["cursorY"]}]
}

move_to_view = {
    "name": "move to view",
    "state_check": [lambda s: s["intent"] == "move"],
    "state_update": [lambda s: {"intent": "view"}]
}

translate_node_left_by = {
    "name": "translate node left by",
    "state_check": [
        lambda s: s["intent"] == "view",
        lambda s: "observation" in s,
        lambda s: "name" in s["observation"],
        lambda s: s["observation"]["name"] == "Key",
        lambda s: "symb" in s["observation"],
        lambda s: s["observation"]["symb"] == 'i',
    ],
    "state_update": [lambda s: {
        "intent": s["intent"],
    }]
}

def rule_applicable(s, rule) -> bool:
    return all(check(s) for check in rule["state_check"])


def rule_update(s, rule):
    return functools.reduce(lambda a, func: func(a), (update for update in rule["state_update"]), s)


class Inside:
    def __init__(self, state_map):
        self.state_map = state_map
        self.transition_rules = [view_to_move, move_to_view]

    def observe(self, inside_observation):
        self.state_map["observation"] = inside_observation

    def proceed(self):
        applicable_rules = [rule for rule in self.transition_rules if
                            rule_applicable(self.state_map, rule)]
        if applicable_rules:
            self.state_map = rule_update(self.state_map, applicable_rules[0])


class Situation:
    def __init__(self, inside, outside_out, outside_in, initial_event_q=None, initial_effect_q=None):
        self.inside = inside
        self.outside_out = outside_out
        self.outside_in = outside_in
        self.event_q = [{"name": "Button-1", "cursorX": 100, "cursorY": 200, "node": 1},
                        {"name": "Key", "symb": "i"}]
        self.effect_q = []
        self.inside_to_outside = {
            "view": lambda s: ViewIntent(s),
            "move": lambda s: MoveIntent(s["node"], s["cursorX"], s["cursorY"]),
        }

    def outside_event_to_inside_observation(self, outside_event):
        return outside_event

    def outside_effect_from_inside_intent(self, inside_intent):
        if inside_intent["intent"] in self.inside_to_outside:
            return Effect(self.inside_to_outside[inside_intent["intent"]](inside_intent))
        raise Exception()

    def step(self):
        self.inside.proceed()
        self.effect_q.append(self.outside_effect_from_inside_intent(self.inside.state_map))
        if self.effect_q:
            self.outside_out.perform(self.effect_q.pop(0))
        outside_event = self.outside_in.poll()
        if outside_event is not None:
            self.event_q.append(outside_event)
        if self.event_q:
            inside_observation = self.outside_event_to_inside_observation(self.event_q.pop(0))
            self.inside.observe(inside_observation)
        self.outside_out.step()


def main():
    tkinter_out = TkInterOutsideOutput(
        dispatcher=ScopedTypeDispatcher(
            {
                ViewIntent: TkInterOutsideOutput.perform_view,
                MoveIntent: TkInterOutsideOutput.perform_move
            }, scope=None))
    tkinter_in = TkInterOutsideIntput(tkinter_out.sketch)
    cmd_out = CmdOutsideOutput(
        dispatcher=TypeDispatcher(
            {
                ViewIntent: CmdOutsideOutput.perform_view,
                MoveIntent: CmdOutsideOutput.perform_move
            }
        )
    )
    sit = Situation(Inside({"intent": "view"}),
                    tkinter_out,
                    tkinter_in
                    )
    while sit.outside_out.alive:
        sit.step()


if __name__ == "__main__":
    main()
