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
        return "Alice"
    return Effect(Move("Bob", n)).on(lambda m: Effect(ContinueGame("Bob", n, m)))


def alice_turn(n: int):
    if n <= 0:
        return "Bob"
    return Effect(Move("Alice", n)).on(lambda m: Effect(ContinueGame("Alice", n, m)))


def game(n: int):
    return alice_turn(n)


@sync_performer
def perfect(dispatcher, move: Move):
    return max(1, move.sticks)


@sync_performer
def play_game(dispatcher, continue_game: ContinueGame):
    if isinstance(continue_game.move, int):
        if continue_game.player == "Alice":
            return bob_turn(continue_game.sticks - continue_game.move)
        elif continue_game.player == "Bob":
            return alice_turn(continue_game.sticks - continue_game.move)
        return Effect(Error(Exception(f"Player is not Alice or Bob, but {continue_game.player}")))
    return Effect(Error(Exception(f"Move is not an int: {continue_game.move}")))


def main():
    dispatcher_perfect: ComposedDispatcher = ComposedDispatcher([
        TypeDispatcher({Move: perfect,
                        ContinueGame: play_game}),
        base_dispatcher])

    sync_perform(dispatcher_perfect, game(7).on(success=lambda res: print(res)))
