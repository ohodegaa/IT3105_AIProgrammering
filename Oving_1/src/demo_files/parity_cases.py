import tensorflow as tf
from utils import tflowtools as tft
from utils.caseman import Caseman
from utils.gann2 import Gann2

"""
Parameter_0: number of bits
Parameter_1: double (When double=True, a 2-bit vector is the target, with bit 0
                    indicating even parity and bit 1 indicating odd parity)
"""
data_set = tft.gen_all_parity_cases(10)

cman = Caseman(lambda: data_set, 0.1, 0.1)

dims = [10, 16, 6, 2]

prefered_accuracy = 0.95

print("Acuuracy should be: ", prefered_accuracy)

gann = Gann2(dims, cman, top_k=1,
             loss_function=lambda labels, predictions: tf.losses.mean_squared_error(labels=labels,
                                                                                    predictions=predictions),
             output_activation_function=tf.nn.softmax,
             hidden_activation_function=tf.nn.leaky_relu,
             optimizer=tf.train.AdamOptimizer,
             learning_rate=0.0035,
             minibatch_size=10,
             init_weight_range=(-0.1, 0.1)
             )

sess = tft.gen_initialized_session()

gann.add_summary(0, "weights", ["avg"])
gann.add_summary(2, "output", ["avg"])

gann.run(sess, 250, validation_interval=10)
tft.close_session(sess)
