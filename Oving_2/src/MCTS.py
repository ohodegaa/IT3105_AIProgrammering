from random import choices, choice
import math
from game_state import GameState
import time

class Node:
    def __init__(self, move=None, parent=None, game: GameState = None):
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.state = game.get_state()

        self.untried_moves = game.get_moves()
        self.player = game.player

    def select_child(self):
        s = sorted(self.children,
                   key=lambda child: child.wins / child.visits +
                                     math.sqrt(2 * math.log(self.visits) / child.visits))[-1]
        return s

    def add_child(self, move, game: GameState):
        child = Node(move, self, game=game)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result):
        self.wins += result
        self.visits += 1

    def get_statistics(self):
        return self.wins / self.visits

    def get_training_case(self):
        feature = []
        target = []
        _target = [[0.0 for _ in range(len(_row))] for _row in self.state]

        for child in self.children:
            row, col = child.move
            _target[row][col] = child.get_statistics()

        for i in range(len(self.state)):
            feature.extend(self.state[i][:])
            target.extend(_target[i])
        return feature, target

    def __str__(self):
        return "Node: move: " + str(self.move)


class MonteCarlo:

    def __init__(self, game: GameState, max_rollouts: int):
        self.max_rollouts = max_rollouts
        self.root_node = Node(game=game)
        self.game = game

    def run(self, get_next=None):
        for r in range(self.max_rollouts):
            node = self.root_node
            game = self.game.clone()
            leaf_node = self.search_tree(node, game)
            node = self.expand_node(leaf_node, game)
            self.rollout(game, get_next)
            self.backpropagate(node, game)

    def get_training_case(self):
        return self.root_node.get_training_case()

    def get_best_move(self):
        return sorted(self.root_node.children, key=lambda c: c.visits)[-1].move  # should it be visits here or stat?

    def backpropagate(self, node, game):
        while node is not None:
            node.update(game.get_result(node.player))
            node = node.parent

    def rollout(self, game: GameState, get_best=None):
        while len(game.get_moves()) > 0:
            if get_best is not None:
                linear_state = game.get_linear_state()
                next_move_index = get_best(linear_state)
                next_move = game.index_to_move(next_move_index)
                if next_move in game.get_moves():
                    game.do_move(next_move)
                else:
                    game.do_move(choice(game.get_moves()))
            else:
                game.do_move(choice(game.get_moves()))

    def expand_node(self, node: Node, game: GameState):
        if len(node.untried_moves) > 0:
            move = choice(node.untried_moves)
            game.do_move(move)
            node = node.add_child(move, game)
        return node

    def search_tree(self, root_node, game: GameState):
        node = root_node
        while node.untried_moves == [] and node.children != []:
            node = node.select_child()
            game.do_move(node.move)
        return node
