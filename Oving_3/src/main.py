import math

from MCTS import MonteCarlo
from hex import Hex
from board import Board
from replay_buffer import ReplayBuffer
from anet import ANET
import utils.tflowtools as tft
import time
import tensorflow as tf


def setup_game():
    n = ask_for_int("Hvilken størrelse på brettet ønsker du? (N) ")
    num_games = ask_for_int("Hvor mange runder ønsker du å spille? (G) ")
    verbose = ask_for_boolean("Vil du vise detaljene for alle trekkene i alle spillene? (y/n)")
    starting_player = ask_for_int(
        "Hvilke spiller skal starte hvert spill? (P): \n0. Velg tilfeldig \n1. Spiller 1 \n2. Spiller 2\n")
    max_rollouts = ask_for_int("Antall simuleringer per faktiske trekk? ")
    return n, num_games, verbose, starting_player, max_rollouts


def ask_for_int(msg):
    x = input(msg)
    return int(x)


def ask_for_boolean(msg):
    x = input(msg)
    return x == "y"


def display_move(player, move, next_state):
    print("Spiller", player, "valgte", move)


def display_results(results):
    print("Player 1: ", results.count(1) / len(results))
    print("Player 2: ", results.count(2) / len(results))


def display_winning_player(player):
    print("\nPlayer {} wins".format(player))
    print("\n=======================================\n")


def init_anet(size, buffer):
    return ANET(
        [size + 1, 34, 45, size],
        buffer=buffer,
        batch_size=20
    )


def main():
    # n, num_games, verbose, starting_player, max_rollouts = setup_game()
    n, num_games, verbose, starting_player, max_rollouts = 5, 200, False, 1, 100
    results = []
    game_num = 1
    viewer = None

    ## NN
    buffer = ReplayBuffer(vfrac=0.1, tfrac=0.1, size=40)
    anet = init_anet(n ** 2, buffer)
    train_interval = 2
    saving_interval = 50
    while num_games >= game_num:
        game = Hex(n, starting_player)
        next_root = None
        # viewer = Board(game)
        while game.get_moves():
            mc = MonteCarlo(game, max_rollouts, next_root)
            mc.run(lambda _input: anet.predict(_input))
            case = mc.get_training_case()
            buffer.push(case)
            next_root = mc.get_best_move()
            game.do_move(next_root.move)
            if viewer:
                viewer.do_move(next_root.move, game.player)
        if game_num % train_interval == 0:
            anet.train_model()
            anet.evaluate()
        if game_num % saving_interval == 0:
            anet.save_to_file("anet/model_step_{0}.h5".format(game_num))
        if game.get_result(game.player) == 1:
            results.append(game.player)
        game_num += 1

    if viewer:
        viewer.persist()


if __name__ == '__main__':
    main()

"""

_2 = time.time()
print(_2 - _1)
_1 = time.time()

============================
        if verbose:
            print("Spill nr. {}\n".format(game_num))
        game = Hex(n, starting_player)
        if verbose:
            viewer = Board(game)
        while game.get_moves():
            mc = MonteCarlo(game, max_rollouts)
            root_node = mc.run()
            
            game.do_move(root_node.move)
            if verbose:
                viewer.do_move(root_node.move, game.player)
                display_move(game.player, root_node.move, game.state)

        if game.get_result(game.player) == 1:
            results.append(game.player)
        game_num += 1
        if verbose:
            display_winning_player(game.player)"""
