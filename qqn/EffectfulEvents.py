from effect import Effect, sync_performer, TypeDispatcher, sync_perform

from qqn.GraphMachine import setdictkey, proceed, anim_orientation, view_to_move, has_happened_in_state

example_state = {
    "id": 1,
    "observation": []
}

example_events = [{"name": "click_cell",
                   # "relevant": has_happened_in_state,
                   "update": lambda s, e: setdictkey(s, "hit_node", e["hit_node"]),
                   "hit_node": "node1"},
                  {"name": "drag",
                   # "relevant": has_happened_in_state,
                   "update": lambda s, e: setdictkey(setdictkey(s, "position_x", e["pointer_x"]), "position_y",
                                                     e["pointer_y"]),
                   "pointer_x": 10,
                   "pointer_y": 10,
                   },
                  {"name": "drag",
                   "relevant": has_happened_in_state,
                   "update": lambda s, e: setdictkey(setdictkey(s, "position_x", e["pointer_x"]), "position_y",
                                                     e["pointer_y"]),
                   "pointer_x": 50,
                   "pointer_y": 50,
                   },
                  {"name": "release",
                   # "relevant": has_happened_in_state,
                   "update": lambda s, e: s
                   }
                  ]


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


def main():
    trace = simple_event_chain()
    sync_perform(
        TypeDispatcher({
            DrawIntent: simple_draw,
            DisplayIntent: print_state
        }),
        draw_effect({"id": 1}, simple_observation_src))
    print(trace)


if __name__ == "__main__":
    main()
