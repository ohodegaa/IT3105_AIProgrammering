import copy
from random import choice

max_number_of_players = 2


class GameState:

    def __init__(self, starting_player=1):
        if not starting_player or starting_player not in range(1, max_number_of_players + 1):
            self.player = choice(range(1, max_number_of_players + 1))
        else:
            self.player = 1 if starting_player == 2 else 2

    def do_move(self, move: tuple):
        raise NotImplementedError("You must implement this function")

    def get_moves(self):
        raise NotImplementedError("You must implement this function")

    def get_result(self, p):
        raise NotImplementedError("You must implement this function")

    def clone(self):
        return copy.deepcopy(self)

    def get_next_player(self):
        return 1 if self.player == 2 else 2
