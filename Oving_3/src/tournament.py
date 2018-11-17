from board import Board
from hex import Hex
import os
from anet import ANET
import itertools


class Tournament:

    def __init__(self, k, g):
        self.num_games = g
        self.num_players = k
        self.models = {}

    def fetch_game_models(self, dir_path="anet"):
        models = {}
        for filename in os.listdir(dir_path):
            if filename.endswith(".h5"):
                model = ANET.new_from_file(filename)
                key = self.filename_to_step(filename)
                models[key] = model
            else:
                continue
        self.models = models

    def filename_to_step(self, filename):
        after_step = filename.split("step_")[1]
        step = after_step.split(".h5")[0]
        return int(step)

    def get_games(self):
        return itertools.combinations(self.models.keys(), 2)


def main():
    # n, num_games, verbose, starting_player, max_rollouts = setup_game()
    n, num_games, verbose, starting_player, max_rollouts = 5, 100, False, 1, 100

    results[]

    game_num = 1
    viewer = None
    tournament = Tournament(4, 25)

    while num_games >= game_num:
        game = Hex(n, starting_player)
        viewer = Board(game)
        while game.get_moves():

            linear_state = game.get_linear_state()
            next_move_index = get_best([game.get_next_player()] + linear_state)
            next_move = game.index_to_move(next_move_index)
            if next_move in game.get_moves():
                game.do_move(next_move)
            else:
                wrong_preds += 1
                game.do_move(choice(game.get_moves()))

            game.do_move(next_move)
            if viewer:
                viewer.do_move(next_move, game.player)

    if viewer:
        viewer.persist()


if __name__ == '__main__':
    main()
