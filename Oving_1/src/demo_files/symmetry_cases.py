import tensorflow as tf
from utils import tflowtools as tft
from utils.caseman import Caseman
from utils.gann2 import Gann2

"""
Parameter_0: length of the bit vector
Parameter_1: number of vectors
"""
data_set = tft.gen_symvect_dataset(101, 2000)

cman = Caseman(lambda: data_set, 0.1, 0.1)

dims = [101, 150, 50, 2]

prefered_accuracy = 0.99

print("Acuuracy should be: ", prefered_accuracy)

gann = Gann2(dims, cman,
             loss_function=lambda labels, predictions: tf.losses.mean_squared_error(labels=labels,
                                                                                    predictions=predictions),
             output_activation_function=tf.nn.softmax,
             hidden_activation_function=tf.nn.leaky_relu,
             optimizer=tf.train.AdamOptimizer,
             learning_rate=0.00003,
             minibatch_size=10,
             init_weight_range=(-0.1, 0.1)
             )

sess = tft.gen_initialized_session()

gann.add_layer_summary(0, "weights", ["avg"])
gann.add_layer_summary(2, "output", ["avg"])

gann.run(sess, 500, validation_interval=10)
tft.close_session(sess)
