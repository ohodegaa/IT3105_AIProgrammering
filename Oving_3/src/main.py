from MCTS import MonteCarlo
from hex import Hex
from board import Board
from replay_buffer import ReplayBuffer
from utils.gann import Gann
import utils.tflowtools as tft
import time


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


def init_gann(size, buffer):
    return Gann(
        layer_sizes=[size, 14, 28, size],
        caseman=buffer,
        top_k=1
    )


def main():
    # n, num_games, verbose, starting_player, max_rollouts = setup_game()
    n, num_games, verbose, starting_player, max_rollouts = 5, 50, False, 1, 100
    results = []
    save_interval = 50
    game_num = 1
    viewer = None

    ## NN
    buffer = ReplayBuffer()
    gann = init_gann(n * n, buffer)
    sess = tft.gen_initialized_session()

    while num_games >= game_num:
        game = Hex(n, starting_player)
        viewer = Board(game)
        while game.get_moves():
            mc = MonteCarlo(game, max_rollouts)
            mc.run(lambda _input: gann.get_best(sess, _input))
            case = mc.get_training_case()
            buffer.push(case)
            next_move = mc.get_best_move()
            game.do_move(next_move)
            viewer.do_move(next_move, game.player)
        gann.run(sess, epochs=100, do_testing=False, show_interval=5)
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
