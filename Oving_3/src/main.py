from MCTS import MonteCarlo
from hex import Hex
from board import Board


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


def main():
    # n, num_games, verbose, starting_player, max_rollouts = setup_game()
    n, num_games, verbose, starting_player, max_rollouts = 5, 1, True, 1, 200
    results = []
    game_num = 1
    while num_games >= game_num:
        if verbose:
            print("Spill nr. {}\n".format(game_num))
        game = Hex(n, starting_player)
        viewer = None
        if verbose:
            viewer = Board(game)
        while game.get_moves():
            mc = MonteCarlo(game, max_rollouts)
            move = mc.run()
            game.do_move(move)
            if verbose:
                viewer.do_move(move, game.player)
                display_move(game.player, move, game.state)

        if game.get_result(game.player) == 1:
            results.append(game.player)
        game_num += 1
        if verbose:
            display_winning_player(game.player)
            print("\n=======================================\n")
    if viewer: viewer.persist()


if __name__ == '__main__':
    main()
