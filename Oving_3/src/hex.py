from game_state import GameState
from path_finder import PathFinder

max_number_of_players = 2


class Hex(GameState):
    CELL_EMPTY = 0
    PLAYER_TOP = -1
    PLAYER_LEFT = 1

    def __init__(self, board_size=5, starting_player=1):
        super(Hex, self).__init__(starting_player)

        self.state = [[self.CELL_EMPTY for _ in range(board_size)] for _ in range(board_size)]
        self.board_size = board_size

    def do_move(self, move: tuple):

        moving_player = self.get_next_player()
        self.state[move[0]][move[1]] = moving_player
        self.player = moving_player

    def get_moves(self):
        moves = []
        if self.get_result(self.player):
            return []
        for i in range(len(self.state)):
            for j in range(len(self.state[i])):
                if self.state[i][j] == 0:
                    moves.append((i, j))
        return moves

    def get_state(self):
        state_copy = []
        for row in self.state[:]:
            state_copy.append(row[:])
        return state_copy

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
