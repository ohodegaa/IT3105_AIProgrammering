import tensorflow as tf


class Gann2:

    def __init__(self, layer_sizes, caseman, learning_rate=.1):
        self.learning_rate = learning_rate
        self.layer_sizes = layer_sizes
        self.caseman = caseman
        self.input = None


    def build_network(self):
        tf.reset_default_graph()
        self.input = tf.placeholder(tf.float32, shape=[None, self.layer_sizes[0]], name="Input")
        in_vars = self.input
        in_size = self.layer_sizes[0]

        for i, out_size in enumerate(self.layer_sizes[1:]):
