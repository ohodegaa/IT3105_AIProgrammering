import tkinter as tk
from hex import Hex
import time

CIRCLE_SIZE = 30
COLORS = ["#eeeeff", "red", "green", "yellow", "orange"]
WINDOW_BORDER = 20


class Board:

    def __init__(self, hex_game, size=None):
        self.root = tk.Tk()
        self.root.title("Hex game with MCST")
        self.circles = {}
        self.size = size if size else max([len(row) for row in hex_game.state])
        window_size = (self.size * 2 - 1) * CIRCLE_SIZE + WINDOW_BORDER * 2
        self.canvas = tk.Canvas(self.root, width=window_size, height=window_size, bg="white", highlightthickness=0)

        self.init_board(hex_game.state)
        self.update_board()

    def init_board(self, hex_state):
        for (i, row) in enumerate(hex_state):  # y
            for (j, cell) in enumerate(row):  # x
                fill = COLORS[cell]
                self.draw_circle(j + 1, i + 1, fill)

    def update_board(self):
        self.canvas.pack()
        self.root.update()

    def draw_circle(self, x, y, fill=None):
        offset_x = (abs(self.size - y) + (x - 1) * 2) * CIRCLE_SIZE + WINDOW_BORDER
        offset_y = (y - 1) * CIRCLE_SIZE + WINDOW_BORDER
        _id = self.canvas.create_oval(offset_x, offset_y, offset_x + CIRCLE_SIZE,
                                      offset_y + CIRCLE_SIZE,
                                      width=2, fill=fill)
        self.circles[(x, y)] = _id
        self.update_board()

    def do_move(self, move, player: int):
        row = move[0]
        col = move[1]
        self.canvas.itemconfigure(self.circles[(col, row)], fill=COLORS[player])
        self.update_board()

    def persist(self):
        self.root.mainloop()
