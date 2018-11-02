import copy
from random import choice
from board import Board

max_number_of_players = 2


class Hex:

    def __init__(self, board_size=5, starting_player=1):
        row_ranges = [x for x in range(1, board_size + 1)] + [y for y in range(board_size - 1, 0, -1)]
        self.state = [[0 for _ in range(i)] for i in row_ranges]

        if not starting_player or starting_player not in range(1, max_number_of_players + 1):
            self.player = choice(range(1, max_number_of_players + 1))
        else:
            self.player = 1 if starting_player == 2 else 2
        self.board = Board(self)

    def do_move(self, move: tuple, show=False):
        next_player = self.get_next_player()
        self.state[move[0]][move[1]] = next_player
        if show:
            self.board.do_move(move, next_player)
        self.player = 1 if self.player == 2 else 2

    def get_moves(self):
        moves = []
        for row in self.state:
            for cell in row:
                if cell > 0:
                    moves.append(cell)

    def clone(self):
        return copy.deepcopy(self)

    def get_result(self, p):
        pass

    def get_next_player(self):
        return 1 if self.player == 2 else 2
