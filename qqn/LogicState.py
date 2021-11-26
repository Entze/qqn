obj = {
    "name": "Susi",
    "age": 16,
    # "arm": "arm_1"
}

age_is_16 = {
    "pre_proc": "id",
    "actual_check": ("age", "is", 16),
    "or_continue": []
}

main_rule = {
    "pre_proc": "id",
    "actual_check": ("name", "is", "Susi"),
    "or_continue": [{"post_proc": "id",
                     "next_rule": age_is_16}]
}

query = {
    "rule": main_rule,
    "obj": obj,
    "status": "not checked"
}

func_bib = {
    "id": lambda o: o,
    "is": lambda o, c, s: o[c] == s
}

pre_proc_func = {
    "id": "id"
}


def take_pred(actual_check):
    match actual_check:
        case (category, modus, similar):
            return lambda o: func_bib[modus](o, category, similar)


def step_query(queried):
    new_obj = func_bib[pre_proc_func[queried["rule"]["pre_proc"]]](queried["obj"])

    # actual_check performed
    if not take_pred([queried["rule"]["actual_check"]])(new_obj):
        return queried.update({"status": "failed now"})

    # no further rule to be checked
    if not queried["rule"]["or_continued"]:
        return queried.update({"status": "accepted now"})

    # prepare next rule check
    next_rules = queried["rule"]["or_continued"]
    next_rule = next_rules.pop()

    # one strategy to nest continuation rules
    if next_rule["or_continued"]:
        next_rules = next_rule["or_continued"] + next_rules

    # construct updated query object, that has been processed for one step
    next_rule.update({"or_continue": next_rules})
    next_obj = func_bib[next_rule["post_proc"]](new_obj)
    return queried.update({"rule": next_rule,
                           "obj": next_obj})


rule_step = {
    "state_check": lambda q: take_pred(q["rule"]["actual_check"])(func_bib[pre_proc_func[q["rule"]["pre_proc"]]](q["obj"])),
    "update_func": lambda q:
        q.update({"rule": q["rule"]["or_continued"].pop().update({"or_continue": q["rule"]["or_continued"]})})
}

rule_query = {
    "pre_proc": "id",
    "actual_check": ("state_checks", "checks", "obs"),
    "or_continue": [{"post_proc": "update_func",
                     "next_rule": rule_step}]
}




def main():
    pass
    # print(proceed(state1, anim_orientation, energy=1))


if __name__ == "__main__":
    main()
