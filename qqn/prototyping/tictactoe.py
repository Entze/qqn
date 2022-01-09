import torch
from torch import tensor, Tensor

# state: tensor([
# [0, 1, 2],
# [3, 4, 5],
# [6, 7, 8],
# [cp]
# ])
# x:  1
# o: -1
# b:  0

# action: tensor([x,y])
# x:
# +-------+-------+-------+
# | [0,0] | [1,0] | [2,0] |
# +-------+-------+-------+
# | [0,1] | [1,1] | [2,1] |
# +-------+-------+-------+
# | [0,2] | [1,2] | [2,2] |
# +-------+-------+-------+
from qqn.agentmodels.testsuite import test

lines = [
    # horizontal
    ((0, 0), (1, 0), (2, 0)),
    ((0, 1), (1, 1), (2, 1)),
    ((0, 2), (1, 2), (2, 2)),
    # vertical
    ((0, 0), (0, 1), (0, 2)),
    ((1, 0), (1, 1), (1, 2)),
    ((2, 0), (2, 1), (2, 2)),
    # diagonals
    ((0, 0), (1, 1), (2, 2)),
    ((2, 0), (1, 1), (0, 2))
]


def tic_tac_toe_transition(state: Tensor, action: Tensor):
    assert tic_tac_toe_action_islegal(state, action)
    x, y = action.tolist()
    cp = state[3][0]
    new_state: Tensor = state.clone().detach()
    new_state[y, x] = cp
    new_state[3][0] = cp * -1
    return new_state


def tic_tac_toe_state_value_xplayer(state: Tensor) -> Tensor:
    return 1 - tic_tac_toe_state_value_oplayer(state)


def tic_tac_toe_state_value_oplayer(state: Tensor) -> Tensor:
    for line in lines:
        (c1x, c1y), (c2x, c2y), (c3x, c3y) = line
        if torch.abs(state[c1y, c1x] + state[c2y, c2x] + state[c3y + c3x]) >= 3.0:
            return (state[c1y, c1x] == p2).int()

    return tensor(0.5)


def tic_tac_toe_action_islegal(state: Tensor, action: Tensor):
    x, y = action.tolist()
    return (0 <= x <= 2 and 0 <= y <= 2) and state[y, x] == _


def tic_tac_toe_state_isfinal(state: Tensor):
    pass
    if not torch.any(state[0:2] == _):
        return True

    for line in lines:
        (c1x, c1y), (c2x, c2y), (c3x, c3y) = line
        if torch.abs(state[c1y, c1x] + state[c2y, c2x] + state[c3y + c3x]) >= 3.0:
            return True


p1 = 1
p2 = -1
_ = 0
e = _



def pretty_tic_tac_toe(state: Tensor):
    cp = state[3][0]
    cp_sym = 'x' if cp == 1 else 'o'
    s: str = f'Current player: {cp_sym}'
    for y in (0, 1, 2):
        s += "\n+---+---+---+\n|"
        for x in (0, 1, 2):
            val = state[y, x].item()
            sym = ' ' if val == _ else ('x' if val == p1 else 'o')
            s += ' ' + sym + ' |'

    s += "\n+---+---+---+\n"
    return s


initial_state = tensor([
    [_, _, _],
    [_, _, _],
    [_, _, _],
    [p1, e, e]
])
alpha = 1.



test(initial_state=initial_state,
     nr_of_actions=nr_of_actions,
     transition=transition,
     state_value=state_value,
     action_islegal=action_islegal,
     state_isfinal=state_isfinal,
     update_belief=update_belief,
     min_estimation_value=-1,
     max_estimation_value=5,
     alpha=alpha,
     traces=traces,
     progressbar=True)
