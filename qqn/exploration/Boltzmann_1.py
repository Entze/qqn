import functools
import operator

import random
from typing import List

init_state = []


def get_actions(state):
    return [True, False]


def transition(state, action):
    return state + [action]


def proceed(gas, state, pol, introspect_fn=lambda s: s):
    if gas == 0:
        return [(introspect_fn(state), "gas empty")]
    possible_acts = get_actions(state)
    selected_act = pol(possible_acts, state)
    next_state = transition(state, selected_act)
    continued = proceed(gas - 1, next_state, pol, introspect_fn)
    return [(introspect_fn(state), selected_act)] + continued


def head_select(list):
    return list[0]


def last_select(list: List):
    return list[len(list) - 1]


def rand_select(list):
    return random.choice(list)


def head_pol(list, state):
    return list[0]


def random_pol(list, state):
    return random.choice(list)


def select_max_idx(list: List):
    idx = 0
    curr = None
    for i, value in enumerate(list):
        if curr is None:
            curr = value
            idx = i
        elif value > curr:
            curr = value
            idx = i
    return idx


def foldl(func, acc, xs):
    return functools.reduce(func, xs, acc)


def foldable_state_value(foldl_fn, eval_single_elem_fn):
    def folder(state):
        return foldl(foldl_fn, 0, [eval_single_elem_fn(e) for e in state])

    return folder


def state_val_pol(select_fn, state_val_fn):
    def pol(act_list, state):
        state_values = [state_val_fn(transition(state, act)) for act in act_list]
        selected_idx = select_fn(state_values)
        return act_list[selected_idx]

    return pol


def test():
    print("list-head-pol: ")
    print(proceed(5, [True], head_pol))
    print("list-head-pol, introspceted by action_value: ")
    print(proceed(5, [True], head_pol, lambda s: {"state": s, "action-value": {True: 1, False: 0}[last_select(s)]}))
    print("random-pol: ")
    print(proceed(5, [True], random_pol))
    print("list-head-pol, introspceted by action_value: ")
    print(proceed(7, [True], random_pol, lambda s: {"state": s, "action-value": {True: 1, False: 0}[last_select(s)]}))
    print("state_val_pol mit select_max_idx, state_val_sum {True: 1, False: 0}")
    print(proceed(5, [True],
                  state_val_pol(select_max_idx, foldable_state_value(operator.add, lambda e: {True: 1, False: 0}[e]))))

test()