import tkinter as tk
from math import cos, sin, radians


class Board:
    COLORS = ["#eeeeff", "red", "green", "yellow", "orange"]
    WINDOW_BORDER = 20
    LENGTH = 20
    XOFF = 20
    YOFF = 20
    SCALE = 20

    def __init__(self, hex_game, size=None):
        self.root = tk.Tk()
        self.root.title("Hex game with MCST")
        self.polygons = {}
        self.size = size if size else len(hex_game.state)
        window_size = self.size * 3 * self.LENGTH * sin(radians(60)) + self.WINDOW_BORDER * 2
        self.canvas = tk.Canvas(self.root, width=window_size, height=window_size, bg="white", highlightthickness=0)

        self.init_board(hex_game.state)
        self.update_board()

    def init_board(self, hex_state):
        for (i, row) in enumerate(hex_state):  # y
            for (j, cell) in enumerate(row):  # x
                fill = self.COLORS[int(cell)]
                self.draw_polygon(j, i, fill)

    def update_board(self):
        self.canvas.pack()
        self.root.update()

    def draw_polygon(self, col, row, fill=None):
        start_x = (col + row / 2) * self.LENGTH * 2 * sin(radians(60)) + self.WINDOW_BORDER
        start_y = row * self.LENGTH * 2 * sin(radians(48)) + self.WINDOW_BORDER
        angle = 60
        points = []
        for i in range(6):
            end_x = start_x + self.LENGTH * sin(radians(angle * i))
            end_y = start_y + self.LENGTH * cos(radians(angle * i))
            points.extend([start_x, start_y])
            start_x = end_x
            start_y = end_y
        # ref: http://eric_rollins.home.mindspring.com/phex/hex.py.txt
        _id = self.canvas.create_polygon(points, fill=fill, width=2, outline="black")
        self.polygons[(col, row)] = _id
        self.update_board()

    def do_move(self, move, player: int):
        row = move[0]
        col = move[1]
        self.canvas.itemconfigure(self.polygons[(col, row)], fill=self.COLORS[player])
        self.update_board()

    def persist(self):
        self.root.mainloop()
