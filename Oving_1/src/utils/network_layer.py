import tensorflow as tf
import numpy as np


class NetworkLayer():

    def __init__(self, an_network, input_size, input_variable, output_size, index,
                 init_weight_range=(-.1, .1), activation_function=tf.nn.leaky_relu):
        self.an_network = an_network  # The Artifical neuron network
        self.input_size = input_size  # Number of neurons feeding into this layer
        self.input_variable = input_variable  #
        self.output_size = output_size  # Number of neurons in this layer
        self.index = index  # The index of this layer
        self.name = "Layer_" + str(self.index)  # The name of the layer
        self.init_weight_range = init_weight_range  # The range the random variables can choose between
        self.activation_function = activation_function  # F.ex: tf.nn.leaky_relu
        self.weights = None
        self.biases = None
        self.output_variable = None
        self.build_layer()

    # Builds a layer in the network
    def build_layer(self):
        w_0, w_1 = self.init_weight_range
        self.weights = tf.Variable(np.random.uniform(w_0, w_1, size=(self.input_size, self.output_size)),
                                   name=self.name + '_weight', dtype=tf.float64)
        self.biases = tf.Variable(np.random.uniform(w_0, w_1, size=self.output_size),
                                  name=self.name + '_bias', dtype=tf.float64)
        linear = tf.matmul(self.input_variable, self.weights) + self.biases

        self.output_variable = self.activation_function(linear, name=self.name + '_output')
        self.an_network.add_layer(self)

    def get_var(self, type):
        return {
            'input': self.input_variable,
            'output': self.output_variable,
            'weights': self.weights,
            'biases': self.biases
        }[type]

    def gen_summary(self, var_name, spec):
        var = self.get_var(var_name)
        base_name = self.name + '_' + var_name
        with tf.name_scope('summary_'):
            if ('avg' in spec):
                avg = tf.reduce_mean(var)
            if 'avg' in spec:
                return tf.summary.scalar(base_name + '/avg/', avg)
            if 'max' in spec:
                return tf.summary.scalar(base_name + '/max/', tf.reduce_max(var))
            if 'min' in spec:
                return tf.summary.scalar(base_name + '/min/', tf.reduce_min(var))
            if 'hist' in spec:
                return tf.summary.histogram(base_name + '/hist/', var)
