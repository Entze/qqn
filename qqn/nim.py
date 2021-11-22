from typing import Callable, Any, Dict, Collection, Optional, List, Tuple

import eff


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


seven_sticks: State = State(7)


class Move(eff.ects):
    ret: Callable
    move: Callable[[Player, State, Callable], Any]  # int should be a transition function (Callable[[State], State]


class Game:
    def __init__(self, player1: Player, player2: Player, initial_state: State):
        self.player1: Player = player1
        self.player2: Player = player2
        self.state: State = initial_state

    def play(self):
        return self.player1turn()

    def player1turn(self):
        if self.state.number_of_sticks <= 0:
            return Move.ret(self.player2)
        self.state.number_of_sticks -= Move.move(self.player1, self.state)
        return self.player2turn()

    def player2turn(self):
        if self.state.number_of_sticks <= 0:
            return Move.ret(self.player1)
        self.state.number_of_sticks -= Move.move(self.player2, self.state)
        return self.player1turn()


def perfect(player: Player, state: State) -> int:
    return max(1, state.number_of_sticks % 4)


def stupid(player: Player, state: State) -> int:
    if player.name == 'Alice':
        return 1
    if player.name == 'Bob':
        if state.number_of_sticks <= 3:
            return state.number_of_sticks
        return 1


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
    with Move(move=perfect, ret=identity):
        print(game.play())
    with Move(move=makegametree, ret=winner):
        print(game.play())
