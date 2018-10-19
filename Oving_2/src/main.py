from nim import Nim

def setup_game():
    N = input("Hvilke størrelse på brettet ønsker du? (N) ")
    K = input("Hvor mange brikker er lov å ta vekk? (K) ")
    return N, K

N, K = setup_game()
nim = Nim(N, K)

while True:
    print("Brikker igjen: ", nim.states[-1][1])
    print("Spiller", nim.player, "din tur")
    while True:
        move = int(input("Hvor mange brikker vil du fjerne? "))
        legal_moves = Nim.next_legal_moves(nim, nim.states)
        if move in legal_moves:
            break
        print("Ikke et lovlig trekke, prøv igjen.")

    Nim.add_state(nim, move)
    Nim.winning(nim)
    if nim.winner:
        break
    Nim.switch_player(nim)

    print("Liste med alle states: ", Nim.get_state(nim))
    print("De neste lovlige statene ", Nim.next_legal_states(nim, nim.states[-1]))
    print('\n')