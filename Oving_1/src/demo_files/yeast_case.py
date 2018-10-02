import tensorflow as tf
from utils import tflowtools as tft
from utils.caseman import Caseman
from utils.gann2 import Gann2
from utils.preparation_data import yeast

data_set = yeast()

cman = Caseman(lambda: data_set, 0.1, 0.1)

dims = [8, 100, 100, 9]

prefered_accuracy = 0.90

print("Accuracy should be: ", prefered_accuracy)

gann = Gann2(dims, cman, top_k=1,
             loss_function=lambda labels, predictions: tf.losses.mean_squared_error(labels=labels,
                                                                                    predictions=predictions),
             output_activation_function=tf.nn.softmax,
             hidden_activation_function=tf.nn.leaky_relu,
             optimizer=tf.train.GradientDescentOptimizer,
             learning_rate=0.1,
             minibatch_size=10,
             init_weight_range=(-0.1, 0.1)
             )

sess = tft.gen_initialized_session()

gann.add_summary(0, "weights", ["avg"])
gann.add_summary(1, "output", ["avg"])

gann.run(sess, 400, validation_interval=10)
tft.close_session(sess)
