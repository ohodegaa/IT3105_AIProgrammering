import os
import itertools
from anet import ANET
from board import Board
from hex import Hex
from MCTS import get_feature
from random import choice
import pprint


class Tournament:

    def __init__(self, num_games, against_random=False):
        self.num_games = num_games
        self.models = {}
        self.results = {}
        self.wins = {}
        self.random = against_random

    def fetch_game_models(self, dir_path="anet"):
        for filename in os.listdir(dir_path):
            if filename.endswith(".h5"):
                model = ANET.new_from_file(dir_path + "/" + filename)
                key = self.filename_to_step(filename)
                self.models[key] = model
            else:
                continue

    def run_tournament(self, path="topp", verbose=False):
        self.fetch_game_models(path)
        games = self.get_games()
        wrong_moves = 0
        total_moves = 0
        viewer = None
        for p1, p2 in games:
            for i in range(self.num_games):
                game = Hex(5, 1)
                if verbose and i == self.num_games - 1:
                    viewer = Board(game)
                while len(game.get_moves()) > 0:
                    if self.random and game.player == Hex.PLAYER_LEFT:
                        next_move = choice(game.get_moves())
                    else:
                        model = self.models[p1 if game.player == Hex.PLAYER_TOP else p2]
                        next_move = ANET.predict(get_feature(game.get_state(), game.player == Hex.PLAYER_LEFT),
                                                 model=model)
                        next_move = (next_move[1], next_move[0]) if game.player == Hex.PLAYER_LEFT else next_move
                        if next_move not in game.get_moves():
                            wrong_moves += 1
                            next_move = choice(game.get_moves())
                    if i == self.num_games - 1 and viewer is not None:
                        viewer.do_move(next_move, game.player)

                    total_moves += 1
                    game.do_move(next_move)
                res_p1 = game.get_result(Hex.PLAYER_TOP)
                res_p2 = game.get_result(Hex.PLAYER_LEFT)
                if res_p1 == 0 and res_p2 == 0:
                    raise Exception
                if res_p1 == 1 and res_p2 == 1:
                    raise Exception

                winning_model = p1 if res_p1 == 1 else p2
                loosing_model = p2 if res_p2 == 0 else p1
                if self.results.get((winning_model, loosing_model)) is not None:
                    self.results[(winning_model, loosing_model)] += 1
                else:
                    self.results[(winning_model, loosing_model)] = 1
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.results)
        if total_moves > 0:
            pp.pprint("Wrong moves (%): " + str(wrong_moves / total_moves))

    def filename_to_step(self, filename):
        after_step = filename.split("step_")[1]
        step = after_step.split(".h5")[0]
        return int(step)

    def get_games(self):
        if self.random:
            return [(mod, "random") for mod in self.models.keys()]
        return itertools.combinations(self.models.keys(), 2)
