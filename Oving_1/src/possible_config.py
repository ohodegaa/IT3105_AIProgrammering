import tensorflow as tf
import tflowtools as tft
from utils.preparation_data import glass, iris, mnist, wine, yeast


"""
Many of the loss function needs different named parameters, but they often refer
to the same, namely the target and the output. By providing it as an array
function to the gann we can easily control any additional parameters and
make the gann highly general in case of which loss function to use
"""


loss_functions = {
    "mean_squared_error": lambda labels, predictions: tf.losses.mean_squared_error(labels=labels, predictions=predictions),
    "absolute_difference": lambda labels, predictions: tf.losses.absolute_difference(labels=labels, predictions=predictions),
    "softmax_cross_entropy": lambda labels, predictions: -tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=predictions),
    "sigmoid_cross_entropy": lambda labels, predictions: tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=predictions),
    "hinge_loss": lambda labels, predictions: tf.losses.hinge_loss(labels=labels, logits=predictions),
    "huber_loss": lambda labels, predictions: tf.losses.huber_loss(labels=labels, predictions=predictions),
    "pairwise_squared_error": lambda labels, predictions: tf.losses.mean_pairwise_squared_error(labels=labels, predictions=predictions),
}

output_activation_functions = {
    "softmax": tf.nn.softmax,
    "leaky_relu": tf.nn.leaky_relu,
    "relu": tf.nn.relu,
    "relu6": tf.nn.relu6,
    "elu": tf.nn.elu,
    "selu": tf.nn.selu,
    "softplus": tf.nn.softplus,
    "softsign": tf.nn.softsign,
    "sigmoid": tf.nn.sigmoid,
    "tanh": tf.nn.tanh
}

hidden_activation_functions = {
    "softmax": tf.nn.softmax,
    "leaky_relu": tf.nn.leaky_relu,
    "relu": tf.nn.relu,
    "relu6": tf.nn.relu6,
    "elu": tf.nn.elu,
    "selu": tf.nn.selu,
    "softplus": tf.nn.softplus,
    "softsign": tf.nn.softsign,
    "sigmoid": tf.nn.sigmoid,
    "tanh": tf.nn.tanh
}

optimizers = {
    "RMSProp": tf.train.RMSPropOptimizer,
    "gradient_descent": tf.train.GradientDescentOptimizer,
    "adagrad": tf.train.AdagradOptimizer,
    "adadelta": tf.train.AdadeltaOptimizer,
    "momentum": lambda lrate: tf.train.MomentumOptimizer(learning_rate=lrate, momentum=0.5),
    "adam": tf.train.AdamOptimizer,
    "ftrl": tf.train.FtrlOptimizer,
}

case_sets = {
    "autoencoder_all_one_hot_cases": tft.gen_all_one_hot_cases,
    "autoencoder_dense_cases": tft.gen_dense_autoencoder_cases,
    "bit_counter_cases": tft.gen_vector_count_cases,
    "glass_case": glass,
    "iris_case": iris,
    "mnist_cases": mnist,
    "parity_cases": tft.gen_all_parity_cases,
    "segment_counter": tft.gen_segmented_vector_cases,
    "symmetry_cases": tft.gen_symvect_dataset,
    "winequality_case": wine,
    "yeast_case": yeast
}

