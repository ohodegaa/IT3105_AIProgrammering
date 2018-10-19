from nim import Nim
from random import choices, choice
import math


class Node:
    def __init__(self, move=None, parent=None, game: Nim = None):
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0

        self.untried_moves = game.get_moves()
        self.player = game.player

    def select_child(self):
        s = sorted(self.children,
                   key=lambda child: child.wins / child.visits +
                                     math.sqrt(2 * math.log(self.visits) / child.visits))[-1]
        return s

    def add_child(self, move, game: Nim):
        child = Node(move, self, game=game)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result):
        self.wins += result
        self.visits += 1


class MonteCarlo:

    def __init__(self, game: Nim, max_rollouts):
        self.max_rollouts = max_rollouts
        self.root_node = Node(game=game)
        self.game = game

    def run(self):

        for r in range(self.max_rollouts):
            node = self.root_node
            game = self.game.clone()
            leaf_node = self.search_tree(node, game)
            node = self.expand_node(leaf_node, game)
            self.rollout(game)
            self.backpropagate(node, game)

        return sorted(self.root_node.children, key=lambda c: c.visits)[-1].move

    def backpropagate(self, node, game):
        while node is not None:
            node.update(game.get_result(node.player))
            node = node.parent

    def rollout(self, game: Nim):
        while game.get_moves():
            game.do_move(choice(game.get_moves()))

    def expand_node(self, node: Node, game: Nim):
        if node.untried_moves:
            move = choice(node.untried_moves)
            game.do_move(move)
            node = node.add_child(move, game)
        return node

    def search_tree(self, root_node, game):
        node = root_node
        while node.untried_moves == [] and node.children != []:
            node = node.select_child()
            game.do_move(node.move)
        return node


def main():
    N = 14
    K = 4
    max_rollouts = 1500
    starting_player = 1
    results = []
    num_games = 100
    while num_games > 0:
        nim = Nim(N, K, max_rollouts, starting_player)
        while nim.get_moves():
            # print("Game state: " + str(nim.state))
            if nim.get_next_player() == 1:
                print("player 1 ", end="")
                mc = MonteCarlo(nim, max_rollouts)
                move = mc.run()
            else:
                print("player 2 ", end="")
                mc = MonteCarlo(nim, max_rollouts)
                move = mc.run()
            print("moved: " + str(move))
            nim.do_move(move)
        print("==========================\n\n")

        if nim.get_result(nim.player) == 1:
            results.append(nim.player)
            # print(str(nim.player) + " won the game")
        # else:
        # print("error")

        # print("Result: ")
        # print(num_games)
        num_games -= 1
    print(results)
    print("Player 1: ", results.count(1) / len(results))
    print("Player 2: ", results.count(2) / len(results))


if __name__ == '__main__':
    main()
