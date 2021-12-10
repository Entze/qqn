from collections import defaultdict
from itertools import repeat
from typing import List, Dict, Optional, Union, Tuple

import torch
from torch import Tensor, tensor, int64

gridworld_actions = ['N', 'O', 'S', 'W']


def transition_gridworld(gridworld_t: Tensor, move: Tensor, player_symb: int = 2) -> Tensor:
    position = current_agent_position(gridworld_t, player_symb)
    new_gridworld_t = torch.clone(gridworld_t)
    x, y = position[0], position[1]
    new_gridworld_t[y, x] = 0
    t = torch.where(move == 0, tensor([0, -1]),
                    torch.where(move == 1, tensor([1, 0]), torch.where(move == 2, tensor([0, 1]), tensor([-1, 0]))))
    npos = position + t
    nx, ny = npos[0], npos[1]
    new_gridworld_t[ny, nx] = player_symb
    return new_gridworld_t


def make_gridworld(grid: List[List[str]], start: Tuple[int, int] = (0, 0),
                   symbol_dict: Optional[Dict[str, int]] = None):
    if symbol_dict is None:
        symbol_dict = defaultdict(lambda: 'X')
    symbol_dict = {' ': 0, '#': 1, 'P': 2} | symbol_dict
    height = len(grid)
    x, y = start
    g = grid.copy()
    g[y][x] = 'P'
    return gridworld_to_tensor(g, symbol_dict)


def current_agent_position(gridworld_t: Tensor, player_symb: int = 2) -> Tensor:
    return torch.squeeze((gridworld_t == player_symb).nonzero()).flip([0])


def legal_actions(gridworld_t: Tensor, irrtraversable_symb: int = 1, player_symb: int = 2) -> Tensor:
    height = gridworld_t.size(dim=0)
    width = gridworld_t.size(dim=1)
    allowed_moves = tensor([True, True, True, True])

    position = current_agent_position(gridworld_t, player_symb)

    at_zero = position == 0
    allowed_moves[3] = allowed_moves[3] and not at_zero[0]
    allowed_moves[0] = allowed_moves[0] and not at_zero[1]

    at_edge = torch.logical_or(position == (height - 1), position == (width - 1))
    allowed_moves[1] = allowed_moves[1] and not at_edge[0]
    allowed_moves[2] = allowed_moves[2] and not at_edge[1]

    x = position[0]
    y = position[1]

    allowed_moves[0] = allowed_moves[0] and (gridworld_t[y - 1, x] != irrtraversable_symb)
    allowed_moves[2] = allowed_moves[2] and (gridworld_t[y + 1, x] != irrtraversable_symb)

    allowed_moves[1] = allowed_moves[1] and (gridworld_t[y, x + 1] != irrtraversable_symb)
    allowed_moves[3] = allowed_moves[3] and (gridworld_t[y, x - 1] != irrtraversable_symb)

    return allowed_moves


def display_gridworld_tensor(gridworld_t: Tensor, symbol_dict: Dict[int, str]):
    height = gridworld_t.size(dim=0)
    width = gridworld_t.size(dim=1)
    max_width = max(len(s) for s in symbol_dict.values())
    d = '   ' + ' '.join(str(w).center(max_width) for w in range(width)) + ' \n'
    d += '  +' + (''.join(('-' * max_width) + '+') * width) + '\n'
    i = 0
    for rows in gridworld_t:
        d += f'{i} |'
        i += 1
        for cell in rows:
            text = symbol_dict.get(cell.item(), "X").center(max_width)
            d += text
            d += '|'
        d += '\n  +' + (''.join(('-' * max_width) + '+') * width) + '\n'
    print(d)


def display_gridworld_list(gridworld_list: List[List[str]]):
    height = len(gridworld_list)
    width = len(gridworld_list[0])
    max_width = max(max(len(cell) for cell in rows) for rows in gridworld_list)
    d = '+' + (''.join(('-' * max_width) + '+') * width) + '\n'
    for y in range(height):
        d += '|'
        for x in range(width):
            cell = gridworld_list[y][x]
            d += cell.center(max_width)
            d += '|'
        d += '\n+' + (''.join(('-' * max_width) + '+') * width) + '\n'
    print(d)


def display_gridworld(grid_world, symbol_dict: Optional[Union[Dict[int, str], Dict[str, int]]] = None):
    if symbol_dict is None:
        symbol_dict = {}
    if symbol_dict and isinstance(next(k for k in symbol_dict.keys()), str):
        symbol_dict = invert_symbol_dict(symbol_dict)
    symbol_dict = {0: ' ', 1: '#', 2: 'P'} | symbol_dict
    if isinstance(grid_world, Tensor):
        g = torch.clone(grid_world)
        display_gridworld_tensor(g, symbol_dict)
    elif isinstance(grid_world, list):
        g = grid_world.copy()
        display_gridworld_list(g)


def invert_symbol_dict(symbol_dict: Dict[int, str]) -> Dict[str, int]:
    return {v: k for k, v in symbol_dict.items()}


def gridworld_to_tensor(gridworld_list: List[List[str]],
                        symbol_dict: Optional[Union[Dict[int, str], Dict[str, int]]] = None):
    if isinstance(symbol_dict, dict):
        if isinstance(next(k for k in symbol_dict.keys()), int):
            inverse_symbol_dict = {' ': 0, '#': 1, 'P': 2} | invert_symbol_dict(symbol_dict)
        else:
            inverse_symbol_dict = {' ': 0, '#': 1, 'P': 2} | symbol_dict
    else:
        inverse_symbol_dict = {' ': 0, '#': 1, 'P': 2}

    return tensor([[inverse_symbol_dict.get(cell, 4) for cell in rows] for rows in gridworld_list], dtype=int64)
