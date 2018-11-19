from random import choices, choice
import math
from game_state import GameState
from hex import Hex
import numpy as np
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
                   key=lambda child: child.get_statistics() +
                                     math.sqrt(2 * math.log(self.visits) / (1 + child.visits)))[-1]
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
        return float(self.wins / self.visits)

    """
    def get_training_case(self):
        feature = []
        feature.append(self.player)
        target = []
        _target = np.zeros((len(self.state), len(self.state)))

        for child in self.children:
            row, col = child.move
            _target[row][col] = child.get_statistics()

        # normalization
        tmax, tmin = _target.max(), _target.min()
        _target = (_target - tmin) / (tmax - tmin)

        for i in range(len(self.state)):
            feature.extend(self.state[i][:])
            target.extend(_target[i])

        return np.array(feature), np.array(target)
    """

    def get_training_case(self):
        feature = get_feature(self.state, self.player == Hex.PLAYER_LEFT)
        target = get_target(self.state, self.children, self.player == Hex.PLAYER_LEFT)

        return [feature, target]

    def __str__(self):
        return "Node: move: " + str(self.move)


def get_target(state, children, transpose=False):
    target = np.zeros((len(state), len(state)), dtype=np.float64)

    for child in children:
        i, j = child.move
        target[i][j] = 1 - float(child.get_statistics())
    target = target.transpose() if transpose else target
    return target


def get_feature(state, invert=False):
    feature = np.zeros((len(state), len(state), 2))
    _state = state if not invert else state.transpose()
    for i in range(len(_state)):
        for j in range(len(_state)):
            _state[i][j] = Hex.PLAYER_LEFT if _state[i][j] == Hex.PLAYER_TOP else (
                Hex.PLAYER_TOP if _state[i][j] == Hex.PLAYER_LEFT else Hex.CELL_EMPTY)
            p1 = int(_state[i][j] == Hex.PLAYER_TOP)
            p2 = int(_state[i][j] == Hex.PLAYER_LEFT)
            free = int(_state[i][j] == Hex.CELL_EMPTY)
            feature[i][j] = [p1, p2]

    return feature


class MonteCarlo:

    def __init__(self, game: GameState, max_rollouts: float, root: Node = None):
        self.max_rollouts = max_rollouts
        self.root_node = root if root is not None else Node(game=game)
        self.game = game
        self.wrong_preds = 0
        self.total_preds = 0

    def run(self, predict=None):
        begin = time.time()
        while time.time() - begin < self.max_rollouts:
            node = self.root_node
            game = self.game.clone()
            if len(game.get_moves()) <= 0:
                return
            leaf_node = self.search_tree(node, game)
            node = self.expand_node(leaf_node, game)
            self.rollout(game, predict)
            self.backpropagate(node, game)
        if self.total_preds > 0:
            return
            print("Wrong by AI: {}/{} ---> {}".format(self.wrong_preds, self.total_preds,
                                                      float(self.wrong_preds / self.total_preds)))

    def get_training_case(self):
        return self.root_node.get_training_case()

    def get_best_move(self):
        return sorted(self.root_node.children, key=lambda c: c.get_statistics())[-1]

    def backpropagate(self, node, game):
        while node is not None:
            node.update(game.get_result(node.player))
            node = node.parent

    def rollout(self, game: GameState, predict=None):
        while len(game.get_moves()) > 0:
            if predict is not None:
                feature = get_feature(game.get_state(), game.player == Hex.PLAYER_LEFT)
                next_move = predict(feature)
                self.total_preds += 1

                next_move = (next_move[1], next_move[0]) if game.player == Hex.PLAYER_LEFT else next_move
                if next_move in game.get_moves():
                    game.do_move(next_move)
                else:
                    self.wrong_preds += 1
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


"""
def generate_linear_feature(state, next_player):
    feature = []
    feature.append(next_player)

    for i in range(len(state[:])):
        feature.extend(state[i][:])

    return np.array(feature)


def generate_feature(state, next_player):
    player1 = np.array(state)
    player2 = np.array(state)
    for i, row in enumerate(player1.tolist()):
        for j, cell in enumerate(row):
            player1[i][j] = 1 if player1[i][j] == -1 else 0
            player2[i][j] = 1 if player2[i][j] == 1 else 0
    player = np.ones((len(state), len(state))) if next_player == 1 else np.zeros(
        (len(state), len(state)))
    feature = np.array([player1, player2, player])
    return feature
"""
