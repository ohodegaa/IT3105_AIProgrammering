import tensorflow as tf


class Gann2:

    def __init__(self, layer_sizes, caseman, learning_rate=.1, init_weight_range=(-.1, .1), hidden_activation_function=tf.nn.relu, output_activation_function=tf.nn.softmax_cross_entropy_with_logits):
        self.learning_rate = learning_rate
        self.layer_sizes = layer_sizes
        self.caseman = caseman
        self.input = None
        self.network_output = None
        self.target = None
        self.init_weight_range = init_weight_range
        self.hidden_activation_function = hidden_activation_function
        self.output_activation_function = output_activation_function
        self.layers = []


    def build_network(self):
        tf.reset_default_graph()
        self.input = tf.placeholder(tf.float32, shape=[None, self.layer_sizes[0]], name="Input")
        input_variable = self.input
        input_size = self.layer_sizes[0]

        for i, output_size in enumerate(self.layer_sizes[1:]):
            layer = NetworkLayer(network=self,
                                 input_size=input_size,
                                 input_variable=input_variable,
                                 output_size=output_size,
                                 index=i,
                                 init_weight_range=self.init_weight_range,
                                 activation_function=self.hidden_activation_function)
            input_variable = layer.output_variable
            input_size = layer.output_size

        self.network_output = layer.output_variable

        if self.output_activation_function is not None:
            self.network_output = self.output_activation_function(self.network_output)

        self.target = tf.placeholder(tf.float64, shape=(None, layer.output_size), name="Target")

    def add_layer(self, layer):
        self.layers.append(layer)