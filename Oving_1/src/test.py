import tensorflow as tf
from utils import tflowtools as tft
from utils.caseman import Caseman
from utils.gann2 import Gann2
import numpy as np
import matplotlib.pyplot as plt

data_set = tft.gen_all_parity_cases(4)

cman = Caseman(lambda: data_set, 0.1, 0.1)

dims = [4, 15, 6, 2]

"""
labels = tf.placeholder(tf.float64, shape=(None, 2), name="Target")
labels = tf.cast(labels, tf.int64)
_labels = tf.argmax(labels, 1)

predictions = tf.placeholder(tf.float64, shape=(None, 2), name="Predictions")
_predictions = tf.cast(predictions, tf.float32)

sess = tf.Session()
feed_dict = {labels: np.array([[0, 1], [1, 0], [1, 0], [0, 1]]),
             predictions: np.array([[0.02, 0.85], [0.76, 0.54], [0.78, 0.78], [0.56, 0.45]])}

is_correct = tf.nn.in_top_k(_predictions, targets=_labels, k=1)
print(sess.run(is_correct, feed_dict=feed_dict))
"""

prefered_accuracy = 0.95

print("Acuuracy should be: ", prefered_accuracy)
gann = Gann2(dims, cman, top_k=1,
             loss_function=lambda labels, predictions: tf.losses.mean_squared_error(labels=labels,
                                                                                    predictions=predictions),
             output_activation_function=tf.nn.softmax,
             hidden_activation_function=tf.nn.leaky_relu,
             optimizer=tf.train.AdamOptimizer,
             learning_rate=0.0033,
             minibatch_size=10
             )

sess = tft.gen_initialized_session()
gann.add_summary(0, "weights", "hist")
gann.add_summary(1, "weights", ["avg", "max", "min"])
gann.add_summary(gann.accuracy)
gann.add_summary(gann.error)
#gann.add_fetched_var(0, "weights")
gann.add_fetched_var(0, ["input", "weights", "output"])
"""
gann.add_summary(2, "output", ["avg", "max", "min"])
"""

# show in hinton:
"""
    input, output i hvert layer, og output
"""

gann.run(sess, 100, validation_interval=20, show_interval=20)
tft.close_session(sess)
