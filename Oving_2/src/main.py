from IT3105_AIProgrammering.Oving_2.src.nim import Nim


def setup_game():
    N = input("Hvilke størrelse på brettet ønsker du? (N) ")
    K = input("Hvor mange brikker er lov å ta vekk i et spill? (K) ")
    G = input("Hvor mange runder ønsker du å spille? (G) ")
    verbose = input("Vil du vise detalajene for alle trekkene i alle spillene? \n1. JA \n2. NEI ")
    P = input("Hvilke spiller skal starte hvert spill? (P): \n1. Spiller 1 \n2. Spiller 2 \n3. Velg tilfeldig ")
    M = input("Antall simuleringer per faktiske trekk? ")
    return N, K, G, verbose, P, M


def verbose(move):
    # Må returnere hvem som vinner spillet også
    print("Spiller", nim.player, "din tur")
    print("Spiller", nim.player, "valgte", move, "steiner: Gjenværende steiner: ", nim.states[-1][1])


def final_statistics():
    # A typical summary (for G = 50) would be a simple statement such as: Player 1 wins 40 of 50 games (80%).
    pass


def first_player():
    pass


def player_move():
    while True:
        move = int(input("Hvor mange brikker vil du fjerne? "))
        legal_moves = Nim.next_legal_moves(nim, nim.states)
        if move in legal_moves:
            break
        print("Ikke et lovlig trekke, prøv igjen.")
    return move

def print_states():
    print("Liste med alle states: ", Nim.get_state(nim))
    print("De neste lovlige statene ", Nim.next_legal_states(nim, nim.states[-1]))

N, K, G, verb, P, M = setup_game()
nim = Nim(N, K)


while True:

    move = player_move()
    print("verbose: ", verb)
    if verb == "1":
        print("HEI!")
        verbose(move)
    Nim.add_state(nim, move)
    Nim.winning(nim)
    if nim.winner:
        break
    Nim.switch_player(nim)


    print('\n')


