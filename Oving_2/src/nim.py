

class Nim():

    def __init__(self, N, K):
        # N = number of sticks on the board
        # K = number of sticks each player can remove

        self.states = [(0, int(N))]
        self.K = int(K)
        self.player = 1
        self.winner = None


    def add_state(self, move):
        new_state = int(self.states[-1][1]) - int(move)
        self.states.append((self.player, new_state))

    def winning(self):
        if int(self.states[-1][1]) == 0:
            print("Spiller", self.player, "vinner!")
            self.winner = True

    def switch_player(self):
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1

    def next_legal_moves(self, state):
        legal_moves = []
        state = state[-1][1]
        for move in range(1, self.K + 1):
            if move <= state:
                legal_moves.append(move)
        return legal_moves

    def next_legal_states(self, state):
        legal_states = []
        player = state[0]
        if player == 1:
            player = 2
        else:
            player = 1
        state = int(state[1])
        for move in range(1, self.K + 1):
            if move <= state:
                new_state = state - move
                legal_states.append((player, new_state))
        return legal_states


    def get_state(self):
        return self.states


