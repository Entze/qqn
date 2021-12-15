def make_prop(triple):
    match triple:
        case (compare_type, check_modus, similar_thing):
            return {"check_modus": check_modus,
                    "compare_type": compare_type,
                    "similar_thing": similar_thing}


def do_state_check(obj, prop_dict):
    if prop_dict["check_modus"] in checkers:
        return checkers[prop_dict["check_modus"]](obj, prop_dict["compare_type"], prop_dict["similar_thing"])


checkers = {
    "is": lambda o, c, s: o[c] == s,
    "greater_than": lambda o, c, s: o[c] > s,
    "one_of": lambda o, c, s: type(o) is list and bool(o)
}

arm_length_smaller_15 = {
    "state_prepare": lambda o: o["arms"],
    "state_check": lambda o: do_state_check(o, make_prop(("List", "one_of", "item"))),
    "conj_rules": None,
}

is_an_adult = {
    "state_prepare": lambda o, r: o,
    "state_check": ("age", "greater_than", 16),
    "conj_rules": []
}

root_rule = {
    "state_prepare": lambda o, r: o,
    "state_check": ("name", "is", "susi"),
    "conj_rules": [is_an_adult]
}

object = {
    "name": "Susi",
    "age": 16,
    # "arm": "arm_1"
}


def check_rule(o, rule):
    new_o = rule["state_prepare"](o, rule)
    if not do_state_check(new_o, make_prop(rule["state_check"])):
        return None
    return (("conj_rules" not in rule or rule["conj_rules"] is None) or all(
        check_rule(new_o, r) for r in rule["conj_rules"])) and (
                   "(disj_rules" not in rule or rule["disj_rules"] is None) or any(
        check_rule(new_o, r) for r in rule["disj_rules"])


is_empty_rule = {
    "state_prepare": lambda o, r: len(o),
    "state_check": ("Object", "is", 0),
}

is_not_empty_rule = {
    "state_prepare": lambda o, r: len(o),
    "state_check": ("Object", "isn't", 0)
}

all_longer_than = {
    "state_prepare": lambda o, r: o,
    "state_check": ("forall", "Elements", "greater", 15)
}
