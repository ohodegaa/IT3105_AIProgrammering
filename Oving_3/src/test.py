from keras.layers import Conv1D, InputLayer, Conv2D
from keras import Sequential
import numpy as np

x_train = np.array([[
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ],
    [
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ],
    [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ],
]])
y_train = np.array([
    [0, 0, 0.3, .4],
    [0.2, 0, .4, .6],
    [0.3, 0.5, 0, 0],
    [0.4, .3, .2, 0]
])

print(x_train.shape)
print(y_train.shape)
model = Sequential()
model.add(Conv2D(filters=2, kernel_size=2, input_shape=(3, 4, 4), padding="same"))
model.compile(loss="mse", optimizer="adam")
print(model.summary())
print("hello")
model.fit(x_train, y_train)
