import copy
from random import choice
import numpy as np

max_number_of_players = 2


class GameState:

    def __init__(self, starting_player=1):
        self.player = starting_player
        self.state = None

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

    def get_state(self):
        return np.copy(self.state)
