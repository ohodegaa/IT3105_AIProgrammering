import tensorflow as tf
from utils import tflowtools as tft
from utils.caseman import Caseman
from utils.gann2 import Gann2
from Oving1.mnist import mnist_basics as mn

"""
Parameter_0: type = "training"
Parameter_1: path =
Parameter_2: unify =
"""
data_set = mn.load_all_flat_cases(
    dir='/Users/olavskogen/Documents/OneDrive/Dokumenter/Skole/2018 Høst/AI Programmering/Innleveringer/Oving1/mnist/')[
           0:6000]
print(list(zip(*data_set)))


def from_int_to_list(input, num):
    return input, [1 if i == num else 0 for i in range(10)]


cman = Caseman(lambda: list(map(lambda el: from_int_to_list(el[0], el[1]), list(zip(*data_set)))), 0.1, 0.1)

dims = [784, 500, 10]

prefered_accuracy = 0.95

print("Acuuracy should be: ", prefered_accuracy)

gann = Gann2(dims, cman,
             loss_function=lambda labels, predictions: tf.losses.mean_squared_error(labels=labels,
                                                                                    predictions=predictions),
             output_activation_function=tf.nn.softmax,
             hidden_activation_function=tf.nn.leaky_relu,
             optimizer=tf.train.AdamOptimizer,
             learning_rate=0.0003,
             minibatch_size=10,
             init_weight_range=(-0.1, 0.1)
             )

sess = tft.gen_initialized_session()

gann.add_layer_summary(0, "weights", ["avg"])
gann.add_layer_summary(1, "output", ["avg"])

gann.run(sess, 10, validation_interval=10)
tft.close_session(sess)
