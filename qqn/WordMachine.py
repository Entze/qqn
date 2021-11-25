'''
view_to_move = {
    "name": ("view", "to", "move"),
    "state_check":
        lambda s:
        make_value_pred(("motive", "is", "view"), s)
        and
        was_informed(("courser", "selects", "node"), s),
    "state_update":
        lambda s:
        update_val(("decision", "is", "move"), s)
}'''

import functools


def set_key_val(store, key, value):
    store[key] = value
    return store


def prepare_content(s, store, content):
    match store:
        case "observation":
            s = functools.reduce(lambda s1, info: s1.update(info),
                                 [obs[content] for obs in s["observation"]],
                                 s["knowledge"])
            return s
    return s


def clear_key(s, key):
    s.update({key:[]})
    return s


interpreter = {
    "check":
        {"is": lambda s, key, value: s[key] == value},
    "informed":
        {"selects": lambda s, actor, selection: any(
            [obs["actor"] == actor and obs["selection"] == selection for obs in s["observation"]])},
    "remember":
        {"with": lambda s, store, memory: s[store].append(s)},
    "update":
        {"is": lambda s, key, value: set_key_val(s, key, value)},
    "extract":
        {"for": lambda s, store, content: prepare_content(s, store, content)},
    # s["knowledge"].update(s[store][content])},
    "clear":
        {"with": lambda s, key, value: set_key_val(s, key, [])}
}

view_to_move_literate = {
    "name": ("view", "to", "move"),
    "state_check": [
        ("check", "motive", "is", "view"),
        ("informed", "courser", "selects", "node")],
    "state_update": [
        # ("remember", "trace", "with", "state"),
        ("update", "motive", "is", "move"),
        ("extract", "observation", "for", "details"),
        ("clear", "observation", "with", "nothing")
    ]
}

move_to_view_literate = {
    "name": ("move", "to", "view"),
    "state_check": [
        ("check", "motive", "is", "move"),
        ("informed", "courser", "releases", "node")],
    "state_update": [
        # ("remember", "trace", "with", "state"),
        ("update", "motive", "is", "view"),
        ("extract", "observation", "for", "details"),
        ("clear", "observation", "with", "nothing")
    ]
}

view_to_insert_literate = {
    "name": ("view", "to", "insert"),
    "state_check": [
        ("check", "motive", "is", "insert"),
        ("informed", "keyboard", "pressed", "button")],
    "state_update": [
        # ("remember", "trace", "with", "state"),
        ("update", "motive", "is", "insert"),
        ("extract", "observation", "for", "details"),
        ("clear", "observation", "with", "nothing")
    ]
}


rules = [view_to_move_literate]

obs1 = {
    "actor": "courser",
    "selection": "node",
    "details": {
        "x": 10,
        "y": 15,
        "node_id": 100
    }
}



state = {
    "motive": "view",
    "observation": [obs1],
    "knowledge": {"old": 3},
    "trace": []
}


def interpret(four_tuple):
    match four_tuple:
        case (mode, thing, relates, signum):
            return lambda s: interpreter[mode][relates](s, thing, signum)


def rule_applicable(s, rule) -> bool:
    return all(interpret(check)(s) for check in rule["state_check"])


def rule_update(s, rule):
    return functools.reduce(lambda s1, func: func(s1),
                            [interpret(update) for update in rule["state_update"]],
                            s)


def main():
    print(rule_applicable(state, view_to_move_literate))
    print(state["motive"])
    print(state["observation"])
    print(state["observation"][0])
    print(state["observation"][0]["details"])
    # state["knowledge"].update(state["observation"][0]["details"])
    print(state)
    # prepare_content(state, "observation", "details")
    #print(state)
    # print(set_key_val(state, "motive", "move"))
    # print(interpreter["update"]["is"](state, "motive", "move"))
    # print(interpreter["remember"]["with"](state, "trace", "state"))
    # print(interpreter["update"]["is"](state, "motive", "move"))
    # ("update", "motive", "is", "move"),
    # ("remember", "trace", "with", "state")
    #print(state)
    interpreter["clear"]["with"](state, "observation", [])
    #print(state)

    #interpret(("extract", "observation", "for", "details"))(state)
    print(state)
    #interpret(("clear", "observation", "with", "nothing"))(state)
    #functools.reduce(lambda s1, func: func(s1),
                     #[   interpret(("extract", "observation", "for", "details")),
                     #    interpret(("clear", "observation", "with", "nothing")),
                     #],
                     #state)
    #prepare_content(state, "observation", "details")
    # print(rule_update(state, view_to_move_literate))
    #print(state)
    # print(proceed(state1, anim_orientation, energy=1))


if __name__ == "__main__":
    main()
