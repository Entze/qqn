from typing import Callable, Any, Dict, Collection, Optional, List, Tuple
from copy import copy

from effect import sync_perform, sync_performer, Effect, TypeDispatcher, Constant, ComposedDispatcher, base_dispatcher


class Player:
    def __init__(self, name: str):
        self.name: str = name

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


alice: Player = Player("Alice")
bob: Player = Player("Bob")


class State:
    def __init__(self, number_of_sticks: int):
        self.number_of_sticks: int = number_of_sticks

    def __copy__(self):
        return State(self.number_of_sticks)


seven_sticks: State = State(7)


class Move(object):
    def __init__(self, player: Player, state: State):
        self.player: Player = player
        self.state: State = state


class ChangeState(object):
    def __init__(self, state: State, move):
        self.state: State = state
        self.move = move


class LogError(object):
    def __init__(self, message: str):
        self.message: str = message


class Game:
    def __init__(self, player1: Player, player2: Player, initial_state: State):
        self.player1: Player = player1
        self.player2: Player = player2
        self.state: State = copy(initial_state)

    def play(self):
        return self.player1turn()

    def player1turn(self):
        if self.state.number_of_sticks <= 0:
            return Effect(Constant(self.player2))
        Effect(Move(self.player1, self.state)).on(
            success=lambda m: Effect(ChangeState(self.state, m)),
            error=lambda e: Effect(LogError(f"Error: {e}"))
        )
        return self.player2turn()

    def player2turn(self):
        if self.state.number_of_sticks <= 0:
            return Effect(Constant(self.player1))
        Effect(Move(self.player2, self.state)).on(
            success=lambda m: Effect(ChangeState(self.state, m)),
            error=lambda e: Effect(LogError(f"Error: {e}"))
        )
        return self.player1turn()


@sync_performer
def perfect(player: Player, state: State) -> int:
    return max(1, state.number_of_sticks % 4)


@sync_performer
def stupid(player: Player, state: State) -> int:
    if player.name == 'Alice':
        return 1
    if player.name == 'Bob':
        if state.number_of_sticks <= 3:
            return state.number_of_sticks
        return 1


@sync_performer
def play_game(state: State, move):
    if isinstance(move, int):
        state.number_of_sticks -= move
        return state
    return None


game: Game = Game(alice, bob, initial_state=seven_sticks)


def identity(x):
    return x


def validmoves(n) -> Collection[int]:
    return frozenset(filter(lambda x: x <= 3, range(n)))


class GameTree:
    def __init__(self, player: Player, subtrees: Optional[List[Tuple[int, Any]]] = None):
        self.kind = "Winner"
        if subtrees is not None:
            self.kind = "Take"
        self.player: Player = player
        self.subtrees: Optional[List[Tuple[int, GameTree]]] = subtrees

    def __repr__(self):
        if self.kind == "Winner":
            return f"Winner({repr(self.player)})"
        if self.kind == "Take":
            return f"Subtrees"
        assert False


def winner(player):
    return GameTree(player)


def makegametree(player, state, cont):
    subgames = map(cont, validmoves(state.number_of_sticks))
    subtrees = zip([1, 2, 3], subgames)
    return GameTree(player, subtrees)


def main():
    perfect_game = Game(alice, bob, seven_sticks)
    perfect_game_rev = Game(bob, alice, seven_sticks)

    stupid_game = Game(alice, bob, seven_sticks)

    dispatcher_perfect = ComposedDispatcher([
        TypeDispatcher({Move: perfect,
                        LogError: print,
                        ChangeState: play_game}),
        base_dispatcher
    ])
    print(sync_perform(dispatcher_perfect, perfect_game.play()))
