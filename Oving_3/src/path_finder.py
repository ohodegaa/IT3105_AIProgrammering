class PathFinder:
    def __init__(self, state, player):
        self.state = state
        self.player = player
        self.visited = []
        self.queue = []

    def is_winner(self):
        for i in range(len(self.state)):
            tup = (0, i) if self.player == 1 else (i, 0)
            if self.state[tup[0]][tup[1]] == self.player:
                # print(tup[0], tup[1])
                if self.is_path([tup]):
                    return True
        return False

    def is_path(self, queue):
        if len(queue) > 0:
            pos = queue.pop()
            self.visited.append(pos)
            if self.winning_pos(pos):
                return True
            queue.extend(self.get_next_positions(pos))
            return self.is_path(queue)
        return False

    def winning_pos(self, pos):
        return pos[0] == len(self.state) - 1 if self.player == 1 else pos[1] == len(self.state) - 1

    def get_next_positions(self, current):
        positions = []
        if current[1] - 1 >= 0:
            positions.append((current[0], current[1] - 1))  # west
        if current[1] + 1 < len(self.state[0]):
            positions.append((current[0], current[1] + 1))  # east
        if current[0] + 1 < len(self.state):
            positions.append((current[0] + 1, current[1]))  # south-east
            if current[1] - 1 >= 0:
                positions.append((current[0] + 1, current[1] - 1))  # south-west
        if current[0] - 1 >= 0:
            positions.append((current[0] - 1, current[1]))  # north-west
            if current[1] + 1 < len(self.state):
                positions.append((current[0] - 1, current[1] + 1))  # north-east
        next_positions = []
        for pos in positions:
            pos_state = self.state[pos[0]][pos[1]]
            if pos_state == self.player and pos not in self.visited:
                next_positions.append(pos)
        return next_positions
