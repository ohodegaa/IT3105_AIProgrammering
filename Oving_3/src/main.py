import math

from MCTS import MonteCarlo
from hex import Hex
from replay_buffer import ReplayBuffer
from anet import ANET
from tournament import Tournament
from keras import optimizers, losses
from MCTS import Node


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
        size=size,
        buffer=buffer,
        batch_size=20,
        optimizer=optimizers.Adagrad(0.005)
    )


def main():
    # n, num_games, verbose, starting_player, max_rollouts = setup_game()
    n, num_games, verbose, starting_player, max_rollouts = 5, 200, False, 1, 0.5
    results = []
    game_num = 1
    viewer = None

    run_tournament = True
    with_training = True
    num_games_tournament = 25
    if run_tournament:
        save_path = "short_topp"
    else:
        save_path = "long_topp"

    ##### CONFIG #####

    buffer_size = 40
    train_interval = 40
    saving_interval = 10
    moves_done = 0
    epochs = 300

    ##################

    buffer = ReplayBuffer(vfrac=0.1, tfrac=0.1, size=buffer_size)
    anet = init_anet(n, buffer)

    if with_training:
        anet.save_to_file(save_path + "/model_step_{0}.h5".format(0))
    game = Hex(n, starting_player)
    ROOT_NODE = Node(game=game)
    while with_training and num_games >= game_num:
        game = Hex(n, starting_player)
        next_root = ROOT_NODE
        # viewer = Board(game)
        print("Game number {}".format(game_num))
        while game.get_moves():
            mc = MonteCarlo(game, max_rollouts, next_root)
            mc.run(lambda _input: ANET.predict(_input, model=anet.model))
            case = mc.get_training_case()
            buffer.push(case)
            next_root = mc.get_best_move()
            game.do_move(next_root.move)
            moves_done += 1

            if viewer:
                viewer.do_move(next_root.move, game.player)
            if moves_done % train_interval == 0:
                buffer.update()
                anet.train_model(epochs)
                anet.run_against_random(num_games=50, game_num=game_num)
        if saving_interval > 0 and game_num % saving_interval == 0:
            anet.save_to_file(save_path + "/model_step_{0}.h5".format(game_num))
            buffer.size += 20
            # train_interval += 5
            # anet.optimizer.lr /= 2
        if game.get_result(game.player) == 1:
            results.append(game.player)
        game_num += 1

    if viewer:
        viewer.persist()

    if run_tournament:
        tournament = Tournament(num_games_tournament)
        tournament.run_tournament(save_path)

    else:
        anet.save_to_file("best_topp/model_2.h5")


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
