

class MonteCarlo:

    def __init__(self, start_state, game, max_rollouts):
        self.expanded_states = [start_state]
        self.wins = []