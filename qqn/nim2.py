import random
from itertools import repeat
from typing import Optional, List, Tuple

from effect import sync_perform, sync_performer, Effect, TypeDispatcher, Constant, ComposedDispatcher, base_dispatcher, \
    Error


class Move(object):
    def __init__(self, player: str, sticks: int):
        self.player = player
        self.sticks = sticks


class ContinueGame(object):
    def __init__(self, player: str, sticks: int, move):
        self.player: str = player
        self.sticks: int = sticks  # State
        self.move = move


def bob_turn(n: int):
    if n <= 0:
        return Effect(Constant("Alice"))
    return Effect(Move("Bob", n)).on(lambda m: Effect(ContinueGame("Bob", n, m)))


def alice_turn(n: int):
    if n <= 0:
        return Effect(Constant("Bob"))
    return Effect(Move("Alice", n)).on(lambda m: Effect(ContinueGame("Alice", n, m)))


def game(n: int):
    return alice_turn(n)


# ======================================================================================================================

@sync_performer
def perfect(dispatcher, move: Move):
    return max(1, move.sticks % 4)


@sync_performer
def play_game(dispatcher, continue_game: ContinueGame):
    if isinstance(continue_game.move, int):
        if continue_game.player == "Alice":
            return bob_turn(continue_game.sticks - continue_game.move)
        elif continue_game.player == "Bob":
            return alice_turn(continue_game.sticks - continue_game.move)
        return Effect(Error(Exception(f"Player is not Alice or Bob, but {continue_game.player}")))
    elif isinstance(continue_game.move, GameTree):
        return continue_game.move
    elif isinstance(continue_game.move, list):
        continuations = []
        for m in continue_game.move:
            if continue_game.player == "Alice":
                continuations.append(bob_turn(continue_game.sticks - m))
            elif continue_game.player == "Bob":
                continuations.append(alice_turn(continue_game.sticks - m))
            else:
                return Effect(Error(Exception(f"Player is not Alice or Bob, but {continue_game.player}")))
        return continuations

    return Effect(Error(Exception(f"Move is not an int: {continue_game.move}")))


# ======================================================================================================================

def valid_moves(n: int):
    return frozenset(range(1, min(4, n + 1)))


@sync_performer
def build_gametree(dispatcher, move: Move):
    effs: List[Effect] = []
    subgames: List[GameTree] = []
    for valid_move in valid_moves(move.sticks):
        effs.append(Effect(ContinueGame(move.player, move.sticks, valid_move)))
    for eff in effs:
        subgames.append(sync_perform(dispatcher, eff))
    subtrees = list(zip([1, 2, 3], subgames))
    return GameTree(move.player, subtrees)


@sync_performer
def end_gametree(dispatcher, winner: Constant):
    return GameTree(winner.result)


class GameTree:

    def __init__(self, player: str, subtrees: Optional[List[Tuple[int, 'GameTree']]] = None):
        self.kind = 'Winner'
        self.player: str = player
        self.subtrees: Optional[List[Tuple[int, GameTree]]] = None
        if subtrees is not None and len(subtrees) > 0:
            self.kind = 'Take'
            self.subtrees = subtrees

    def __repr__(self):
        if self.kind == 'Winner':
            return f"Winner({self.player})"
        assert self.subtrees is not None
        r = f"{self.player} - ( {len(self.subtrees)} subtrees )"
        return r

    def __str__(self):
        return self.__str_depth()

    def __str_depth(self, depth: int = 0) -> str:
        if self.kind == 'Winner':
            return f"Winner({self.player})"
        assert self.subtrees is not None
        tab = "---"
        r = self.player
        for subgame in self.subtrees:
            r += f"\n|\n+-{''.join(repeat(tab, depth))}( {subgame[0]}: [ {subgame[1].__str_depth(depth + 1)} ] )"
        return r


# ======================================================================================================================

class Cheat(object):
    def __init__(self, player: str):
        self.player: str = player


@sync_performer
def report(dispatcher, cheat: Cheat):
    return Effect(Error(Exception(f"{cheat.player} cheated!")))


@sync_performer
def checker(dispatcher: ComposedDispatcher, move: Move):
    next = ComposedDispatcher(dispatcher.dispatchers[1:])
    m: int = sync_perform(next, Effect(Move(move.player, move.sticks)))
    if m in valid_moves(move.sticks):
        return m
    return Effect(Cheat(move.player))


@sync_performer
def bob_cheat(dispatcher, move: Move):
    if move.player == "Alice":
        return max(1, move.sticks % 4)
    return move.sticks


# ======================================================================================================================

class Choose(object):
    pass


@sync_performer
def bob_chooses(dispatcher, move: Move):
    if move.player == "Bob":
        choices = sync_perform(dispatcher, Effect(Choose()))
        moves = []
        for choice in choices:
            if choice:
                moves.append(move.sticks)
            else:
                moves.append(max(1, move.sticks % 4))
        return moves
    return max(1, move.sticks % 4)


@sync_performer
def all_results(dispatcher, choose: Choose):
    return [True, False]


@sync_performer
def coin(dispatcher, choose: Choose):
    return [bool(random.randint(0, 1))]


# ======================================================================================================================

def main():
    dispatcher_perfect: ComposedDispatcher = ComposedDispatcher([
        TypeDispatcher({Move: perfect,
                        ContinueGame: play_game}),
        base_dispatcher])

    dispatcher_gametree: ComposedDispatcher = ComposedDispatcher([
        TypeDispatcher({Move: build_gametree,
                        ContinueGame: play_game,
                        Constant: end_gametree
                        }),
        base_dispatcher])

    dispatcher_check_cheating: ComposedDispatcher = ComposedDispatcher([
        TypeDispatcher({Move: checker}),
        TypeDispatcher({Cheat: report,
                        Move: perfect,
                        ContinueGame: play_game}),
        base_dispatcher
    ])

    dispatcher_allow_cheating: ComposedDispatcher = ComposedDispatcher([
        TypeDispatcher({Move: bob_cheat,
                        ContinueGame: play_game}),
        base_dispatcher
    ])

    dispatcher_caught_cheating: ComposedDispatcher = ComposedDispatcher([
        TypeDispatcher({Move: checker}),
        TypeDispatcher({Cheat: report,
                        Move: bob_cheat,
                        ContinueGame: play_game}),
        base_dispatcher
    ])

    dispatcher_all_results: ComposedDispatcher = ComposedDispatcher([
        TypeDispatcher({Choose: all_results}),
        TypeDispatcher({Move: bob_chooses,
                        ContinueGame: play_game}),
        base_dispatcher
    ])

    print("\n===\nNormal game:")
    sync_perform(dispatcher_perfect, game(7).on(success=lambda res: print(res)))

    print("\n===\nGametree:")
    sync_perform(dispatcher_gametree, game(3).on(success=lambda res: print(res)))

    print("\n===\nNormal game with checking for cheating:")
    sync_perform(dispatcher_check_cheating, game(7).on(success=lambda res: print(res)))

    print("\n===\nCheating game:")
    sync_perform(dispatcher_allow_cheating, game(7).on(success=lambda res: print(res)))

    # print("\n===\nCheating game with checking for cheating:")
    # sync_perform(dispatcher_caught_cheating, game(7).on(success=lambda res: print(res)))

    print("\n===\nAll results:")
    sync_perform(dispatcher_all_results, game(7).on(success=lambda res: print(res)))


if __name__ == "__main__":
    main()
