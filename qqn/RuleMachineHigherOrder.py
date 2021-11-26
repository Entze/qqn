# typealias = SimpleApplicable = (Any, Any, Any)

# Higher-Order-Applicable = (quantifier, SimpleApplicable)

quantifiers = {
    "forall": {
        "consumer": lambda consumable, rule: all(check_rule(elem, rule) for elem in consumable)
    }
}

methods = {
    "equals": lambda obj, specifier, elem: specifier(obj) == elem
}


def field(name):
    return lambda obj: obj[name]


sample_rule = {
    "applicable": (field("name"), "equals", "susi"),
}

sample_obj = {
    "name": "susi"
}

sample_coll = [
    sample_obj,
    sample_obj,
    {}
]


def apply_rule(obj, rule):
    if "type" not in rule or rule["type"] is None or rule["type"] == "simple":
        specifier, method, elem = rule["applicable"]
        matches = method in methods and methods[method](obj, specifier, elem)
        if not "conj_rules"  rule
    elif rule["type"] == "higher":
        pass
