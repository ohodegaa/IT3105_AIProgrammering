import tensorflow as tf
import numpy as np
import math
from utils import tflowtools as tft
import matplotlib.pyplot as plt

from utils.gann_module import Gannmodule

data_set = np.array([
    [[0, 0, 0], [1, 0]],
    [[0, 0, 1], [0, 1]],
    [[0, 1, 0], [1, 0]],
    [[0, 1, 1], [1, 0]],
    [[1, 0, 0], [1, 1]],
    [[1, 0, 1], [1, 0]],
    [[1, 1, 0], [1, 0]],
    [[1, 1, 1], [0, 1]]
])

inputs = [c[0] for c in data_set]
input_1 = tf.placeholder(tf.float32, shape=[None, 3], name="input_1")

targets = [c[1] for c in data_set]
target = tf.placeholder(tf.float32, shape=[None, 2], name="target")


weights_1 = tf.Variable(np.random.uniform(-.1, .1, size=[3, 2]), dtype=tf.float32, name="weights_1")
biases_1 = tf.Variable(np.random.uniform(-.1, .1, size=[1, 2]), dtype=tf.float32, name="biases_1")

output_1 = tf.nn.elu(tf.matmul(input_1, weights_1) + biases_1, name="output_1")


loss = tf.reduce_mean(tf.square(target - output_1))

optimizer = tf.train.GradientDescentOptimizer(0.1)

trainer = optimizer.minimize(loss=loss, name="backprop")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10):
    results = sess.run([trainer, loss, weights_1], feed_dict={input_1: inputs[4:7], target: targets})


