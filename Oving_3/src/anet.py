from keras import Sequential
from keras import optimizers
from keras.layers import Conv2D, Dense
from replay_buffer import ReplayBuffer
from keras.models import load_model
import numpy as np


class ANET:

    def __init__(
            self,
            dims,
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
            model=None,
    ):
        self.dims = dims
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
        self.model = model if model is not None else Sequential()
        self.set_up_layers()

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
        def set_up_layers(self):
            kernel_size = 2
            input_layer = Conv2D(5, kernel_size, input_shape=(3, self.dims[0], self.dims[0]), use_bias=True, padding="same")
            self.model.add(input_layer)
            print(input_layer.input_shape, input_layer.output_shape)
    
            for dim in self.dims[1:-1]:
                layer = Conv2D(dim, kernel_size, use_bias=True, padding="same")
                self.model.add(layer)
                print(layer.input_shape, layer.output_shape)
            output_layer = Dense(units=5, activation="softmax")
            self.model.add(output_layer)
            print(self.model.summary())
    """

    def get_cases(self, cases):
        x_train = []
        y_train = []
        for i in range(len(cases)):
            x_train.append(np.array(cases[i][0]))
            y_train.append(np.array(cases[i][1]))

        return np.array(x_train), np.array(y_train)

    def predict(self, x_test):
        x_test = np.array([x_test])
        prediction = self.model.predict_classes(x_test)[0]
        # print(x_test, prediction)
        return prediction

    def evaluate(self):
        x_test, y_test = self.get_cases(self.buffer.get_testing_cases())
        self.model.evaluate(x_test, y_test, batch_size=self.batch_size, verbose=False)

    def train_model(self):
        sgd = optimizers.RMSprop(lr=0.01)
        self.model.compile(
            optimizer=sgd,
            loss=self.loss,
            metrics=["accuracy"]
        )
        x_train, y_train = self.get_cases(self.buffer.get_training_cases())
        print(self.buffer.get_validation_cases())
        self.model.fit(x_train, y_train, validation_data=self.get_cases(self.buffer.get_validation_cases()),
                       epochs=self.epochs, batch_size=self.batch_size, verbose=True)

    def save_to_file(self, path="anet/model.h5"):
        self.model.save(path)
        print("saved to file " + path)

    def load_from_file(self, path="anet/model.h5"):
        del self.model
        self.model = load_model(path)

    @staticmethod
    def new_from_file(path="anet/model.h5"):
        return load_model(path)
