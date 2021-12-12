import random
from copy import deepcopy
from typing import List

import pyro.distributions as dist
import torch

start_state = [
    ['?', 'o', '?'],
    ['?', 'x', 'x'],
    ['?', '?', '?']
]


def is_valid_move(state, move):
    return state[move["y"]][move["x"]] == '?'


def lukas_is_valid_action(state, action):
    x = torch.remainder(action, 3)
    y = torch.div(action, 3, rounding_mode='floor')
    return state[y, x] == 0


def lukas_action_prior(state):
    action_weights = torch.clone(state)
    action_weights[action_weights != 0] = float('-inf')
    return action_weights


def action_prior(state):
    move = {
        "x": dist.Categorical(logits=torch.zeros(3)).sample(),
        "y": dist.Categorical(logits=torch.zeros(3)).sample()
    }

    if is_valid_move(state, move):
        return move
    return action_prior(state)


def actions_prior(state):
    moves = [{"x": x, "y": y} for x in range(3) for y in range(3)]
    valid_moves = [move for move in moves if is_valid_move(state, move)]
    return valid_moves


symbol = {
    0: "?",
    1: "x",
    2: "o"
}


def transition(state: List[List[str]], move, player:str):
    # sign = symbol[player]
    new_state = deepcopy(state)
    new_state[move["y"]][move["x"]] = player
    return new_state


def diag1(state):
    return [state[i][i] for i in range(3)]


def diag2(state: List):
    n = len(state)
    return [state[i][n - (i + 1)] for i in range(3)]


def has_won(state, player):
    def check(xs: List):
        return len([x for x in xs if x == player]) == len(xs)

    def get_cols(i):
        return [row[i] for row in state]

    possibilities = [
        state[0], state[1], state[2],  # rows
        get_cols(0), get_cols(1), get_cols(2),  # cols
        diag1(state), diag2(state)  # diagonals
    ]
    return any([check(possible) for possible in possibilities])


won_state = [
    ['?', 'o', '?'],
    ['x', 'x', 'x'],
    ['?', '?', '?']
]


def isDraw(state):
    return not has_won(state, 'x') and not has_won(state, 'o')


def utility(state, player):
    if has_won(state, player):
        return 10
    elif isDraw(state):
        return 0
    return -10


def act(state, player):
    return random.choice(actions_prior(state))


def other_player(player):
    return "x" if player == "o" else "o"


def is_complete(state):
    return all(all(cell != "?" for cell in row) for row in state)


def is_terminal(state):
    return has_won(state, 'x') or has_won(state, 'o') or is_complete(state)


def simulate(state, player):
    if is_terminal(state):
        return [state]
    action = act(state, player)
    assert state[action["y"]][action["x"]] == '?'
    next_state = transition(state, action, player)
    next_player = other_player(player)
    return [state] + simulate(next_state, next_player)

def print_state(state):
    rows = []
    for row in state:
        print('|'.join(row))


def test():
    print(start_state)
    print(actions_prior(start_state))
    print(transition(start_state, {"x": 1, "y": 0}, 'o'))
    print(has_won(won_state, 'x'))
    print('-' * 80)
    print_state(start_state)
    print('-' * 80)

    for t in simulate(start_state, 'x'):
        print_state(t)
        print('-' * 80)
    print('-' * 80)
    for t in simulate(start_state, 'o'):
        print_state(t)
        print('-' * 80)



test()

