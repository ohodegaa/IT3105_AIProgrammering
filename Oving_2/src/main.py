from MCTS import MonteCarlo
from nim import Nim


def setup_game():
    n = ask_for_int("Hvilken størrelse på brettet ønsker du? (N) ")
    k = ask_for_int("Hvor mange brikker er lov å ta vekk i et trekk? (K) ")
    num_games = ask_for_int("Hvor mange runder ønsker du å spille? (G) ")
    verbose = ask_for_int("Vil du vise detalajene for alle trekkene i alle spillene? \n1. JA \n2. NEI ")
    starting_player = ask_for_int(
        "Hvilke spiller skal starte hvert spill? (P): \n0. Velg tilfeldig \n1. Spiller 1 \n2. Spiller 2\n")
    max_rollouts = ask_for_int("Antall simuleringer per faktiske trekk? ")
    return n, k, num_games, verbose, starting_player, max_rollouts


def ask_for_int(msg):
    x = input(msg)
    return int(x)


def display_move(player, move, next_state):
    print("Spiller", player, "valgte", move, "steiner. Gjenværende steiner: ", next_state)


def display_results(results):
    print("Player 1: ", results.count(1) / len(results))
    print("Player 2: ", results.count(2) / len(results))


def display_winning_player(player):
    print("\nPlayer {} wins".format(player))


def main():
    n, k, num_games, verbose, starting_player, max_rollouts = setup_game()

    results = []
    game_num = 1
    while num_games >= game_num:
        if verbose:
            print("Spill nr. {}\n".format(game_num))
        nim = Nim(n, k, starting_player)
        while nim.get_moves():
            if nim.get_next_player() == 1:
                mc = MonteCarlo(nim, max_rollouts)
                move = mc.run()
            else:
                mc = MonteCarlo(nim, max_rollouts)
                move = mc.run()
            nim.do_move(move)
            if verbose == 1:
                display_move(nim.player, move, nim.state)

        if nim.get_result(nim.player) == 1:
            results.append(nim.player)
        game_num += 1
        if verbose:
            display_winning_player(nim.player)
            print("\n=======================================\n")

    display_results(results)


if __name__ == '__main__':
    main()
