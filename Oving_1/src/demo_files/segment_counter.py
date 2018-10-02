import tensorflow as tf
from utils import tflowtools as tft
from utils.caseman import Caseman
from utils.gann2 import Gann2

"""
Parameter_0: length of the bit vector
Parameter_1: number of cases to produce
Parameter_2: minimum number of segments in a vector
Parameter_3: maximum number of segments in a vector
Parameter_4: pop_target=True
"""
data_set = tft.gen_segmented_vector_cases(25,1000,0,8)

cman = Caseman(lambda: data_set, 0.1, 0.1)

dims = [25, 15, 9]

prefered_accuracy = 0.95

print("Acuuracy should be: ", prefered_accuracy)

gann = Gann2(dims, cman, top_k=1,
             loss_function=lambda labels, predictions: tf.losses.mean_squared_error(labels=labels,
                                                                                    predictions=predictions),
             output_activation_function=tf.nn.softmax,
             hidden_activation_function=tf.nn.leaky_relu,
             optimizer=tf.train.AdamOptimizer,
             learning_rate=0.003,
             minibatch_size=10,
             init_weight_range=(-0.1, 0.1)
             )

sess = tft.gen_initialized_session()

gann.add_summary(0, "weights", ["avg"])
gann.add_summary(1, "output", ["avg"])

gann.run(sess, 1500, validation_interval=10)
tft.close_session(sess)
