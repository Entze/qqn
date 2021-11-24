import functools

state = {"id": 1}

orientation = {
    "rule1":
        {"state_check": lambda s: s["id"] == 1,
         "update_func": lambda s: [{"id": 2}]},
    "rule2":
        {"state_check": lambda s: s["id"] == 2,
         "update_func": lambda s: [{"id": 1}]}
}


def proceed(current_state, orientation_map, energy=0, trace=[],
            prioritise_state=lambda l1: l1[0],
            prioritise_transition=lambda l2: l2[0]):
    if energy == 0:
        return trace + [current_state]
    update_func = prioritise_state(
        [ori["update_func"] for ori in orientation_map.values() if ori["state_check"](current_state)])
    new_state = prioritise_transition(update_func(current_state))
    return proceed(new_state, orientation_map, energy - 1, trace + [current_state], prioritise_state,
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
move_to_view = [{"name": "release"}]


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
             [apply_all_updates(s, view_to_move, {"id": 2})]}
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
