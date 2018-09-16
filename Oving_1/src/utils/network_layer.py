import tensorflow as tf
import numpy as np


class NetworkLayer():

    def __init__(self, an_network, input_size, input_variable, output_size, index,
                 init_weight_range=(-.1,.1), activation_function=tf.nn.leaky_relu):
        self.an_network = an_network                 # The Artifical neuron network
        self.input_size = input_size                 # Number of neurons feeding into this layer
        self.input_varable = input_variable          #
        self.output_size = output_size               # Number of neurons in this layer
        self.index = index                           # The index of this layer
        self.name = "Layer_" + str(self.index)       # The name of the layer
        self.init_weight_range = init_weight_range   # The range the random variables can choose between
        self.activation_function = activation_function # F.ex: tf.nn.leaky_relu
        self.weights = None
        self.biases = None
        self.output_variable = None
        self.buld_layer()

    # Builds a layer in the network
    def build_layer(self):
        w_0, w_1 = self.init_weight_range
        self.weights = tf.Variable(np.random.uniform(w_0, w_1, size=(self.input_size,
                                    self.output_size)), name=self.name+'_weight')
        self.biases = tf.Variable(np.random.uniform(w_0, w_1, size=self.output_size),
                                                    name=self.name+'_bias')
        self.output_variable = self.activation_function(tf.matmul(self.input, self.weights) + self.biases,
                                               name=self.name + '_output')
        self.an_network.add_layer(self)
