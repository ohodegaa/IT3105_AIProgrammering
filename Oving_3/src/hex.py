from game_state import GameState
from path_finder import PathFinder

max_number_of_players = 2

CELL_EMPTY = 0
PLAYER_TOP = 1
PLAYER_LEFT = 2


class Hex(GameState):

    def __init__(self, board_size=5, starting_player=1):
        super(Hex, self).__init__(starting_player)

        self.state = [[CELL_EMPTY for _ in range(board_size)] for _ in range(board_size)]
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

    def get_result(self, p):
        path_finder = PathFinder(self.state, p)
        return int(path_finder.is_winner())
