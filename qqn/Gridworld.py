from collections import defaultdict
from itertools import repeat
from typing import List, Dict, Optional, Union

import torch
from torch import Tensor, tensor, int64

gridworld_actions = ['N', 'O', 'S', 'W']


def legal_actions(gridworld_t: Tensor, position: Tensor) -> Tensor:
    height = gridworld_t.size(dim=0)
    width = gridworld_t.size(dim=1)
    allowed_moves = tensor([True, True, True, True])

    at_zero = position == 1
    allowed_moves[3] = allowed_moves[3] and not at_zero[0]
    allowed_moves[2] = allowed_moves[2] and not at_zero[1]

    at_edge = torch.logical_or(position == height, position == width)
    allowed_moves[1] = allowed_moves[1] and not at_edge[0]
    allowed_moves[0] = allowed_moves[0] and not at_edge[1]

    x = position[0] - 1
    y = width - position[1]

    allowed_moves[0] = allowed_moves[0] and (gridworld_t[y - 1, x] != 1)
    allowed_moves[2] = allowed_moves[2] and (gridworld_t[y + 1, x] != 1)

    allowed_moves[1] = allowed_moves[1] and (gridworld_t[y, x + 1] != 1)
    allowed_moves[3] = allowed_moves[3] and (gridworld_t[y, x - 1] != 1)

    return allowed_moves


def display_gridworld_tensor(gridworld_t: Tensor, symbol_dict: Dict[int, str]):
    height = gridworld_t.size(dim=0)
    width = gridworld_t.size(dim=1)
    max_width = max(len(s) for s in symbol_dict.values())
    d = '+' + (''.join(('-' * max_width) + '+') * width) + '\n'
    for rows in gridworld_t:
        d += '|'
        for cell in rows:
            text = symbol_dict.get(cell.item(), "X").center(max_width)
            d += text
            d += '|'
        d += '\n+' + (''.join(('-' * max_width) + '+') * width) + '\n'
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


def display_gridworld(grid_world, symbol_dict: Optional[Dict[int, str]] = None):
    if symbol_dict is None:
        symbol_dict = {}
    symbol_dict = {0: ' ', 1: '#', 2: 'P'} | symbol_dict
    if isinstance(grid_world, Tensor):
        display_gridworld_tensor(grid_world, symbol_dict)
    elif isinstance(grid_world, list):
        display_gridworld_list(grid_world)


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

    return tensor([[inverse_symbol_dict.get(cell, 'X') for cell in rows] for rows in gridworld_list], dtype=int64)
