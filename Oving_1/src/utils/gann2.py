import math

import tensorflow as tf
from utils.network_layer import NetworkLayer
import utils.tflowtools as tft
import numpy as np


class Gann2:

    def __init__(self,
                 layer_sizes,
                 caseman,
                 learning_rate=.1,
                 init_weight_range=(-.1, .1),
                 hidden_activation_function=tf.nn.relu,
                 output_activation_function=tf.nn.softmax,
                 cost_function=lambda target, output: tf.losses.mean_squared_error(labels=target, predictions=output),
                 optimizer=tf.train.RMSPropOptimizer,
                 minibatch_size=10
                 ):

        self.learning_rate = learning_rate
        self.layer_sizes = layer_sizes
        self.caseman = caseman
        self.input = None
        self.output = None
        self.target = None
        self.error = None
        self.trainer = None
        self.cost_function = cost_function
        self.optimizer = optimizer
        self.init_weight_range = init_weight_range
        self.hidden_activation_function = hidden_activation_function
        self.output_activation_function = output_activation_function
        self.error_history = []
        self.layers = []
        self.fetched_vars = []
        self.global_training_step = 0
        self.minibatch_size = minibatch_size
        self.summary_vars = []
        self.build_network()

    def build_network(self):
        tf.reset_default_graph()
        self.input = tf.placeholder(tf.float64, shape=[None, self.layer_sizes[0]], name="Input")
        input_variable = self.input
        input_size = self.layer_sizes[0]

        for i, output_size in enumerate(self.layer_sizes[1:]):
            layer = NetworkLayer(an_network=self,
                                 input_size=input_size,
                                 input_variable=input_variable,
                                 output_size=output_size,
                                 index=i,
                                 init_weight_range=self.init_weight_range,
                                 activation_function=self.hidden_activation_function)
            input_variable = layer.output_variable
            input_size = layer.output_size

        self.output = layer.output_variable

        if self.output_activation_function is not None:
            self.output = self.output_activation_function(self.output)

        self.target = tf.placeholder(tf.float64, shape=(None, layer.output_size), name="Target")
        self.setup_training()

    def add_layer(self, layer):
        self.layers.append(layer)

    def setup_training(self):
        self.error = self.cost_function(self.target, self.output)
        self.error = tf.reduce_mean(self.error)
        optimizer = self.optimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error, name="Trainer")

    def train_network(self, sess, cases, epochs=100, continued=False):
        if not continued: self.error_history = []
        for i in range(epochs):
            step = self.global_training_step + i
            total_error = 0
            fetched_vars = [self.error] + [x.output_variable for x in self.layers]
            num_cases = len(cases)
            num_minibatches = math.ceil(num_cases / self.minibatch_size)

            for case_start in range(0, num_cases, self.minibatch_size):
                case_end = min(num_cases, case_start + self.minibatch_size)
                minibatch = cases[case_start:case_end]
                inputs = [c[0] for c in minibatch]
                targets = [c[1] for c in minibatch]
                feed_dict = {self.input: inputs, self.target: targets}
                _, fetched_values, _ = self.run_network([self.trainer],
                                                        fetched_vars=fetched_vars,
                                                        summary_vars=[self.summary_vars],
                                                        feed_dict=feed_dict,
                                                        step=step,
                                                        session=sess)

                print(fetched_values[1][0])
                total_error += fetched_values[0]
            self.error_history.append((step, total_error / num_minibatches))
        self.global_training_step += epochs

    def run_network(self, operators, fetched_vars=None, summary_vars=None, dir="summary", session=None,
                    feed_dict=None, step=1, show_interval=1):
        if session is None:
            print("Dette er feil!")
            sess = tft.gen_initialized_session(dir)
        else: sess = session
        results = sess.run([operators, fetched_vars, summary_vars], feed_dict=feed_dict)

        return results[0], results[1], sess

    def display_fetched_vars(self, fetched_vals, fetched_vars, step=1):
        names = [x.name for x in fetched_vars]
        msg = "Fetched variables at step " + str(step)
        print("\n" + msg)
        fig_index = 0
        for i, val in enumerate(fetched_vals):
            if names:
                print(">>> " + names[i] + " = " + fetched_vals[i])
            if type(val) == np.ndarray and len(val.shape) > 1:
                tft.hinton_plot(val, fig=self.fetched_var_figures[fig_index], title=names[i] + " at step " + str(step))
                fig_index += 1

            else:
                print(val, end="\n\n")
