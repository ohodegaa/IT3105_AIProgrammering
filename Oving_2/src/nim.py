import copy
from random import choice

max_number_of_players = 2


class Nim:

    def __init__(self, N, K, starting_player):
        # N = number of sticks on the board
        # K = number of sticks each player can remove
        # starting_player = starting player
        # max_rollouts = number of simulations done by MCTS

        self.state = int(N)
        if not starting_player or starting_player not in range(1, max_number_of_players + 1):
            self.player = choice(range(1, max_number_of_players + 1))
        else:
            self.player = 1 if starting_player == 2 else 2
        self.K = int(K)
        # self.winner = None

    def do_move(self, move: int):
        self.state = self.state - move
        self.player = 1 if self.player == 2 else 2

    def get_moves(self):
        legal_moves = []
        for move in range(1, min([self.K, self.state]) + 1):
            legal_moves.append(move)
        return legal_moves

    def clone(self):
        return copy.deepcopy(self)

    def get_result(self, p):
        assert self.state == 0
        if self.player == p:
            return 1
        else:
            return 0

    def get_next_player(self):
        return 1 if self.player == 2 else 2


