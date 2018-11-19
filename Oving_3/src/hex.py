from game_state import GameState
from path_finder import PathFinder
import numpy as np

max_number_of_players = 2


class Hex(GameState):
    CELL_EMPTY = 0
    PLAYER_TOP = 1
    PLAYER_LEFT = 2

    def __init__(self, board_size=5, starting_player=1):
        super(Hex, self).__init__(starting_player)

        self.state = np.zeros((board_size, board_size), dtype=np.float64)
        self.board_size = board_size

    def do_move(self, move: tuple):
        self.state[move[0]][move[1]] = self.player
        self.player = self.get_next_player()

    def get_moves(self):
        moves = []
        if self.get_result(self.player):
            return []
        for i in range(len(self.state)):
            for j in range(len(self.state[i])):
                if self.state[i][j] == self.CELL_EMPTY:
                    moves.append((i, j))
        return moves

    def get_result(self, p):
        path_finder = PathFinder(self, p)
        return int(path_finder.is_winner())

    def get_linear_state(self):
        linear_state = []
        for i in range(len(self.state)):
            linear_state.extend(self.state[i])
        return linear_state

    def index_to_move(self, i):
        row = i // len(self.state[0])
        col = i % len(self.state[row])
        return row, col

# bruk -1, 0  og 1 for spiller 1 og 2 og ingen
# max / min for ulike spillere?
#
