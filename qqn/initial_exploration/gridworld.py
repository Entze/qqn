import math

import torch
from torch import tensor


def invert_embedding(embedding):
    return {v: k for k, v in embedding.items()}


def display_t(gw_t, embedding=None):
    if embedding is None:
        embedding = std_embedding
    embedding = {" ": 0,
                 "#": 1} | embedding
    embedding = invert_embedding(embedding)

    height = gw_t.size(dim=0)
    width = gw_t.size(dim=1)

    max_width = max(max(len(s) for s in embedding.values()), int(math.log10(torch.max(gw_t))))
    d = '   ' + ' '.join(str(w).center(max_width) for w in range(width)) + ' \n'
    d += '  +' + (''.join(('-' * max_width) + '+') * width) + '\n'
    i = 0
    for rows in gw_t:
        d += f'{i} |'
        i += 1
        for cell in rows:
            text = embedding.get(cell.item(), str(cell.item())).center(max_width)
            d += text
            d += '|'
        d += '\n  +' + (''.join(('-' * max_width) + '+') * width) + '\n'
    print(d)


def display(gw):
    height = len(gw)
    width = len(gw[0])
    max_width = max(max(len(cell) for cell in rows) for rows in gw)
    d = '   ' + ' '.join(str(w).center(max_width) for w in range(width)) + ' \n'
    d += '  +' + (''.join(('-' * max_width) + '+') * width) + '\n'
    i = 0
    for y in range(height):
        d += f'{i} |'
        i += 1
        for x in range(width):
            cell = gw[y][x]
            d += cell.center(max_width)
            d += '|'
        d += '\n  +' + (''.join(('-' * max_width) + '+') * width) + '\n'
    print(d)


def as_tensor(gw, embedding=None):
    if embedding is None:
        embedding = std_embedding
    embedding = {" ": 0,
                 "#": 1} | embedding
    return tensor([[embedding.get(sym, -1) for sym in row] for row in gw])


def allowed_actions(gw_t, state_t):
    x, y = state_t[1], state_t[2]
    height = gw_t.size(dim=0)
    width = gw_t.size(dim=1)

    actions = tensor([True for _ in range(4)])
    actions[0] = y != 0
    actions[1] = x != (width - 1)
    actions[2] = y != (height - 1)
    actions[3] = x != 0

    def transition(act):
        return {0: lambda: gw_t[y - 1, x],
                1: lambda: gw_t[y, x + 1],
                2: lambda: gw_t[y + 1, x],
                3: lambda: gw_t[y, x - 1]}[act]

    def validate(act, possible):
        if possible:
            return transition(act)() != 1
        return tensor(False)

    return torch.stack([validate(act, possible) for act, possible in enumerate(actions)])

    # actions[0] = actions[0] and gw_t[y - 1, x] != 1




std_embedding = {
    'DN': 10,
    'DS': 11,
    'V': 12,
    'N': 13,
}
