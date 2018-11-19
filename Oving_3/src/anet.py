from random import choice

from keras import Sequential
from keras import optimizers
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Reshape, BatchNormalization

from MCTS import get_feature
from hex import Hex
from replay_buffer import ReplayBuffer
from keras.models import load_model
import numpy as np
import pprint as pp


class ANET:

    def __init__(
            self,
            size,
            buffer: ReplayBuffer,
            input_act="tanh",
            hidden_act="relu",
            output_act="tanh",
            init="uniform",
            epochs=200,
            batch_size=10,
            verbose=True,
            loss="mse",
            optimizer="adam",
    ):
        self.size = size
        self.buffer = buffer
        self.input_act = input_act
        self.hidden_act = hidden_act
        self.output_act = output_act
        self.init = init
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.loss = loss
        self.optimizer = optimizer
        self.model = None
        self.set_up_model()
        self.learning_rate = 0.1

    """
    def set_up_layers(self):
        input_layer = Dense(units=self.dims[1], use_bias=True, input_shape=(self.dims[0],))
        self.model.add(input_layer)

        for dim in self.dims[2:-1]:
            layer = Dense(dim, use_bias=True)
            self.model.add(layer)

        output_layer = Dense(units=25, activation="tanh")
        self.model.add(output_layer)
        print(self.model.summary())

    """

    def set_up_model(self):
        size = 2
        self.model = Sequential([
            Conv2D(12, (size, size), padding='same', activation='sigmoid', input_shape=(self.size, self.size, 2)),
            # Conv2D(6, (size, size), padding='same', activation='sigmoid'),
            # Conv2D(16, (size, size), padding='same', activation='sigmoid'),
            # Conv2D(32, (size, size), padding='same', activation='sigmoid'),
            # Conv2D(12, (self.size, self.size), padding='same', activation='sigmoid'),
            # Conv2D(6, (size, size), padding='same', activation='sigmoid'),
            Conv2D(1, (self.size, self.size), padding='same', activation='sigmoid'),
            # Dropout(0.25),
            Reshape((self.size, self.size)),
        ])

        print(self.model.summary())

    def get_cases(self, cases):
        x_train = []
        y_train = []
        for i in range(len(cases)):
            x_train.append(np.array(cases[i][0]).copy())
            y_train.append(np.array(cases[i][1]).copy())

        return np.array(x_train), np.array(y_train)

    @staticmethod
    def predict(x_test, model=None, show_prediction=False):
        x_test = np.array([x_test]).copy()
        prediction = model.predict(x_test)
        index_max = np.unravel_index(np.argmax(prediction, axis=None), prediction.shape)
        if show_prediction:
            pp.pprint(prediction)
        return index_max[1:]

    """
    def evaluate(self):
        x_test, y_test = self.get_cases(self.buffer.get_testing_cases())

        correct = 0
        print(len(x_test))
        for i in range(len(x_test)):
            print("Prediction: ")
            index_max_pred = ANET.predict(x_test[i], self.model, show_prediction=False)
            index_max_true = np.unravel_index(np.argmax(y_test[i], axis=None), y_test[i].shape)
            pp.pprint(index_max_pred)
            print("Should have been: ")
            pp.pprint(y_test[i])
            pp.pprint(index_max_true)
            if index_max_pred == index_max_true:
                correct += 1
        print("Correct predictions (test): " + str(float(correct / len(x_test))))
        """

    def train_model(self, epochs=None, verbose=False):
        epochs = epochs if epochs is not None else self.epochs
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
        )
        x_train, y_train = self.get_cases(self.buffer.get_training_cases())
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=self.batch_size, verbose=verbose)

    def save_to_file(self, path="anet/model.h5"):
        self.model.save(path)
        print("saved to file " + path)

    def load_from_file(self, path="anet/model.h5"):
        del self.model
        self.model = load_model(path)

    @staticmethod
    def new_from_file(path="anet/model.h5"):
        return load_model(path)

    def run_against_random(self, num_games=25, game_num=0):
        wrong_moves = 0
        total_moves = 0
        games_won = 0
        for _ in range(num_games):
            game = Hex(5, 1)
            while len(game.get_moves()) > 0:
                if game.player == Hex.PLAYER_TOP:  # model player
                    next_move = ANET.predict(get_feature(game.get_state(), game.player == Hex.PLAYER_LEFT),
                                             model=self.model)
                    next_move = (next_move[1], next_move[0]) if game.player == Hex.PLAYER_LEFT else next_move
                    if next_move not in game.get_moves():
                        wrong_moves += 1
                        next_move = choice(game.get_moves())
                else:
                    next_move = choice(game.get_moves())
                game.do_move(next_move)
                total_moves += 1

            res_model = game.get_result(Hex.PLAYER_TOP)
            res_random = game.get_result(Hex.PLAYER_LEFT)
            if res_model == 0 and res_random == 0:
                print("Draw")
                continue
            games_won += res_model

        win_rate = games_won / num_games
        wrong_moves_rate = wrong_moves / total_moves
        pp.pprint("Game {}    : {}".format(game_num, win_rate))
        pp.pprint("Wrong moves: {}".format(wrong_moves_rate))
        return win_rate, wrong_moves_rate
