import tensorflow as tf
import numpy as np
import math
from utils import tflowtools as tft
import matplotlib.pyplot as plt
from utils.caseman import Caseman
from utils.gann_module import Gannmodule


from utils.gann2 import Gann2

data_set = tft.gen_all_parity_cases(2)

cman = Caseman(lambda: data_set, 0.1, 0.1)

dims = [2, 2, 2]
cfuncs = [
    ("mse", lambda target, output: tf.losses.mean_squared_error(labels=target, predictions=output)),
    #("sig_cross_entr", lambda target, output: tf.losses.sigmoid_cross_entropy(multi_class_labels=target, logits=output)),
    #("softmax_corss_entropy", lambda target, output: tf.losses.softmax_cross_entropy(onehot_labels=target, logits=output)),
]

for i in range(len(cfuncs)):
    #for j in range(2, 13):
    print(cfuncs[i][0] + " --> ", end="")
  #  print(str(j))
    #dims[1] = j
    gann = Gann2(dims, caseman=cman, hidden_activation_function=tf.nn.leaky_relu, cost_function=cfuncs[i][1])

    sess = tft.gen_initialized_session()
    gann.train_network(sess, gann.caseman.get_training_cases(), 15)
    sess.close()

    print("Error: " + str(gann.error_history[-1][1]))
    plt.scatter(*zip(*gann.error_history))
    plt.show()

