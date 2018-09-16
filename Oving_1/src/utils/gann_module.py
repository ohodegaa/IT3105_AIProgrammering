import tensorflow as tf
import numpy as np


# A general ann module = a layer of neurons (the output) plus its incoming weights and biases.
class Gannmodule():

    def __init__(self, ann, index, invariable, insize, outsize, activation_function=tf.nn.relu,
                 init_weight_range=(-.1, .1)):
        self.ann = ann
        self.insize = insize  # Number of neurons feeding into this module
        self.outsize = outsize  # Number of neurons in this module
        self.input = invariable  # Either the gann's input variable or the upstream module's output
        self.index = index
        self.name = "Module-" + str(self.index)
        self.activation_function = activation_function
        self.weights = None
        self.biases = None
        self.output = None
        self.init_weight_range = init_weight_range
        self.build()

    def build(self):
        module_name = self.name
        n = self.outsize
        w_0, w_1 = self.init_weight_range
        self.weights = tf.Variable(np.random.uniform(w_0, w_1, size=(self.insize, n)),
                                   name=module_name + '-wgt', trainable=True)  # True = default for trainable anyway
        self.biases = tf.Variable(np.random.uniform(w_0, w_1, size=n),
                                  name=module_name + '-bias', trainable=True)  # First bias vector
        self.output = self.activation_function(tf.matmul(self.input, self.weights) + self.biases,
                                               name=module_name + '-out')
        self.ann.add_module(self)

    def getvar(self, type):  # type = (in,out,wgt,bias)
        return {'in': self.input, 'out': self.output, 'wgt': self.weights, 'bias': self.biases}[type]

    # spec, a list, can contain one or more of (avg,max,min,hist); type = (in, out, wgt, bias)
    def gen_probe(self, type, spec):
        var = self.getvar(type)
        base = self.name + '_' + type
        with tf.name_scope('probe_'):
            if ('avg' in spec) or ('stdev' in spec):
                avg = tf.reduce_mean(var)
            if 'avg' in spec:
                tf.summary.scalar(base + '/avg/', avg)
            if 'max' in spec:
                tf.summary.scalar(base + '/max/', tf.reduce_max(var))
            if 'min' in spec:
                tf.summary.scalar(base + '/min/', tf.reduce_min(var))
            if 'hist' in spec:
                tf.summary.histogram(base + '/hist/', var)
