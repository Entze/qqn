# typealias = SimpleApplicable = (Any, Any, Any)

# Higher-Order-Applicable = (quantifier, SimpleApplicable)
from typing import Dict

quantifiers = {
    "forall": {
        "consumer": all
    }
}

methods = {
    "equals": lambda a, b: a == b,
    "always": lambda _a, _b: True,
    "never": lambda _a, _b: False,
    "idem": lambda a, _: a,
    "fail": lambda _a, _b: None,
    "show_with": lambda a, b: a + b if isinstance(a, str) else str(a) + b,
    "show": lambda _a, b: b,
    "greater_equals": lambda a, b: a >= b,
    "return": lambda _a, b: b,
    "head": lambda a, _b: a[0],
    "apply": lambda a, b: b(a),
    "head_apply": lambda a, b: apply_method("apply", a[0], "self", b),
    "head_by": lambda a, b: apply_method(b[1], a[0], b[0], b[2]),
    "by": lambda a, b: apply_method(b[1], a, b[0], b[2])
}


def apply_method(method, obj, specifier, elem):
    if method in methods:
        method_func = methods[method]
        if specifier is None:
            return method_func(None, elem)
        return method_func(obj[specifier], elem)

    raise Exception(f"Method {method} not found.")


def set_at_key_value(d: Dict, k, v):
    d[k] = v
    return d


sample_rule = {
    "applicable": ("name", "equals", "susi"),
}

sample_rule = set_at_key_value(sample_rule, "self", sample_rule)

sample_higher_rule = {
    "type": "higher",
    "applicable": ("forall", sample_rule),
}

sample_higher_rule = set_at_key_value(sample_higher_rule, "self", sample_higher_rule)

sample_obj_1 = {
    "name": "susi",
    "age": 16,
    "sex": "female"
}

sample_obj_1 = set_at_key_value(sample_obj_1, "self", sample_obj_1)

sample_obj_2 = {
    "name": "robert",
    "age": 26,
    "sex": "male"
}

sample_obj_2 = set_at_key_value(sample_obj_2, "self", sample_obj_2)

sample_obj_3 = {
    "name": "gerti",
    "age": 25,
    "sex": "female"
}

sample_obj_3 = set_at_key_value(sample_obj_3, "self", sample_obj_3)

sample_coll = {

    "Collection": [
        sample_obj_1,
        sample_obj_2,
        sample_obj_3,
    ]
}

is_over_18 = {
    "type": "simple",
    "applicable": ("age", "greater_equals", 18),
}

is_female = {
    "applicable": ("sex", "equals", "female"),
}

is_snack_rule = {
    "type": "simple",
    "name": "Ist eine Schnecke?",
    "applicable": (None, "always", None),
    # "on_success": ("name", "show_with", " ist eine Schnecke!"),
    "on_fail": ("name", "show_with", " ist leider keine Schnecke!"),
    "conj_rules": [is_over_18, is_female],
    "reduce": ("collection", "head_by", ("name", "show_with", " ist eine Schnecke!")),
}


def apply_rule(obj: Dict, rule):
    if "type" not in rule or rule["type"] is None or rule["type"] == "simple":
        specifier, method, elem = rule["applicable"]
        matches = apply_method(method, obj, specifier, elem)
        post = obj.copy()
        post_specifier, post_method, post_elem = "self", "idem", None
        if matches:
            if "on_success" in rule and rule["on_success"] is not None:
                post_specifier, post_method, post_elem = rule["on_success"]
        else:
            if "on_failure" in rule and rule["on_failure"] is not None:
                post_specifier, post_method, post_elem = rule["on_failure"]
            else:
                post_specifier, post_method, post_elem = "self", "fail", None
        post = apply_method(post_method, post, post_specifier, post_elem)
        conj_present = False
        conj_matches = True
        conj_result = None
        if "conj_rules" in rule and rule["conj_rules"]:
            conj_present = True
            conj_results = []
            for conj_rule in rule["conj_rules"]:
                result = apply_rule(post, conj_rule)
                if result is not None:
                    conj_results.append(result)
                else:
                    conj_matches = False
                    break
            if conj_matches:
                conj_result = {"collection": conj_results}
                conj_result["self"] = conj_result

                reduce_specifier, reduce_method, reduce_elem = "self", "idem", None
                if "reduce" in rule and rule["reduce"] is not None:
                    reduce_specifier, reduce_method, reduce_elem = rule["reduce"]
                conj_result = apply_method(reduce_method, conj_result, reduce_specifier, reduce_elem)

        disj_present = False
        disj_result = None
        if "disj_rules" in rule and rule["disj_rules"]:
            for disj_rule in rule["disj_rules"]:
                result = apply_rule(post, disj_rule)
                if result is not None:
                    disj_result = result
                    break

        if not (conj_present or disj_present):
            combined = post
        elif conj_present and not disj_present:
            combined = conj_result
        elif not conj_present and disj_present:
            combined = disj_result
        else:
            assert conj_present and disj_present
            if "combine" not in rule or rule["combine"] is None:
                raise Exception("Rule application produced conjunctive subresult"
                                " and disjunctive subresult, but no way how to combine the subresults."
                                f" Provide a method ({type(conj_present).__name__} -> {type(disj_present).__name__} "
                                "-> Any)"
                                " in the key 'combine'"
                                "." if "name" not in rule else f" in the rule {rule['name']}.")
            combine_specifier, combine_method, combine_elem = rule["reduce"]
            combined = apply_method(combine_method, conj_result, combine_specifier, disj_result[combine_elem])

        return combined
    elif rule["type"] == "higher":
        pass

    raise Exception(f"Unknown rule with type {rule['type']}")


def main():
    print("Who has the name susi?")
    print(apply_rule(sample_obj_1, sample_rule))
    print(apply_rule(sample_obj_2, sample_rule))
    print(apply_rule(sample_obj_3, sample_rule))

    print("Who is a snack?")
    print(apply_rule(sample_obj_1, is_snack_rule))
    print(apply_rule(sample_obj_2, is_snack_rule))
    print(apply_rule(sample_obj_3, is_snack_rule))


if __name__ == "__main__":
    main()
