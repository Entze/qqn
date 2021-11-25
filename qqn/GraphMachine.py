import functools


def proceed(current_state, orientation_map, energy=None, trace=None,
            prioritise_state=lambda l1: l1[0],
            prioritise_transition=lambda l2: l2[0]):
    if trace is None:
        trace = []
    if energy == 0:
        return trace + [current_state]
    applicable_transition_funcs = [ori["update_func"] for ori in orientation_map.values() if
                                   ori["state_check"](current_state)]
    if not applicable_transition_funcs:
        return trace + [current_state]
    update_func = prioritise_state(
        applicable_transition_funcs
    )
    new_states = update_func(current_state)
    assert new_states
    new_state = prioritise_transition(new_states)
    new_energy = None if energy is None else energy - 1

    return proceed(new_state, orientation_map, new_energy, trace + [current_state], prioritise_state,
                   prioritise_transition)


def setdictkey(dict, key, value):
    dict[key] = value
    return dict


view_state = {"id": 1}
move_state = {"id": 2,
              "node": None}
insert_node_state = {"id": 3}
insert_edge_state = {"id": 4}
connect_node_state = {"id": 5,
                      "node_src": None,
                      "node_dst": None}
debounce_state = {"id": 6}

all_states = {
    "view_state": {"id": 1},
    "move_state": {"id": 2,
                   "node": None},
    "insert_node_state": {"id": 3},
    "insert_edge_state": {"id": 4},
    "connect_node_state": {"id": 5,
                           "node_src": None,
                           "node_dst": None},
    "debounce_state": {"id": 6}
}


def has_happened_in_state(s, e):
    return e["name"] in [obs["name"] for obs in s["observation"]]


def has_happened_in_state_qualified_by(q, s, e):
    return e[q] in [obs[q] for obs in s["observation"]]


view_to_move = [{"name": "click_cell",
                 "relevant": has_happened_in_state,
                 "update": lambda s, e: setdictkey(s, "hit_node", e["hit_node"]),
                 "hit_node": None}]
move_to_view = [{
    "name": "release",
    "relevant": has_happened_in_state,
    "update": lambda s, _: s
}]

drag = [{
    "name": "drag",
    "relevant": has_happened_in_state,
    "update": lambda s, e: setdictkey(setdictkey(s, "position_x", e["pointer_x"]), "position_y", e["pointer_y"])
}]

view_to_insert_node = [{
    "name": "pressed key",
    "pressed_key": "i",
    "relevant": lambda s, e: has_happened_in_state(s, e) and has_happened_in_state_qualified_by(
        "pressed_key", s, e),
}]

insert_node = [{
    "name": "click",
    "pointer_x": None,
    "pointer_y": None,
    "relevant": has_happened_in_state,
    "update": lambda s, e: setdictkey(setdictkey(s, "pointer_x", e["pointer_x"]), "pointer_y", e["pointer_y"])
}]

insert_node_to_view = [{
    "name": "pressed key",
    "pressed_key": "i",
    "relevant": lambda s, e: has_happened_in_state(s, e) and has_happened_in_state_qualified_by("pressed_key", s, e)
}]


def all_conditions_met(s, conditions):
    return all(condition["relevant"](s, condition) for condition in conditions)


def apply_all_updates(s, conditions, init):
    return functools.reduce(lambda s1, func: func(s1),
                            [lambda s2: obs["update"](s2, obs) for obs in s["observation"] if
                             all_conditions_met(s, conditions)], init)


anim_orientation = {
    "rule1":
        {"state_check":
             lambda s:
             s["id"] == 1 and all_conditions_met(s, view_to_move),
         "update_func":
             lambda s:
             [apply_all_updates(s, view_to_move, {"id": 2, "observation": []})]},
    "rule2":
        {"state_check":
             lambda s:
             s["id"] == 2 and all_conditions_met(s, move_to_view),
         "update_func":
             lambda s:
             [{"id": 1, "observation": []}]
         },
    "rule3":
        {"state_check":
             lambda s:
             s["id"] == 2 and all_conditions_met(s, drag),
         "update_func":
             lambda s: [apply_all_updates(s, drag, {"id": 2, "hit_node": s["hit_node"], "observation": []})]
         },
    "rule4":
        {"state_check":
             lambda s:
             s["id"] == 1 and all_conditions_met(s, view_to_insert_node),
         "update_func":
             lambda s: [{"id": 3, "observation": []}]
         },
    "rule5":
        {"state_check":
             lambda s:
             s["id"] == 3 and all_conditions_met(s, insert_node),
         "update_func":
             lambda s: [apply_all_updates(s, insert_node, {"id": 3, "observation": []})]
         },
    "rule6":
        {"state_check":
             lambda s:
             s["id"] == 3 and all_conditions_met(s, insert_node_to_view),
         "update_func":
             lambda s: [{"id": 1, "observation": []}]
         }
    # "idem":
    #    {"state_check":
    #         lambda _: True,
    #     "update_func":
    #         lambda s: [setdictkey(s, "observation", [])]
    #     }
}

state1 = {
    "id": 1,
    "observation": [{"name": "click_cell",
                     "update": lambda s, e: setdictkey(s, "hit_node", e["hit_node"]),
                     "hit_node": "node1"}]
}


def main():
    print(proceed(state1, anim_orientation, energy=1))


if __name__ == "__main__":
    main()
