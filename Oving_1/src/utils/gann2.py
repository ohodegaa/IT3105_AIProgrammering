import math

import tensorflow as tf
from utils.network_layer import NetworkLayer
import utils.tflowtools as tft
import numpy as np
import matplotlib.pyplot as plt
from utils.image_plotter import draw_scatter, draw_error_vs_validation


class Gann2:

    def __init__(self,
                 layer_sizes,
                 caseman,
                 learning_rate=.1,
                 init_weight_range=(-.1, .1),
                 hidden_activation_function=tf.nn.relu,
                 output_activation_function=tf.nn.softmax,
                 loss_function=lambda labels, predictions: tf.losses.mean_squared_error(labels=labels,
                                                                                        predictions=predictions),
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
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.init_weight_range = init_weight_range
        self.hidden_activation_function = hidden_activation_function
        self.output_activation_function = output_activation_function
        self.error_history = []
        self.validation_history = []
        self.validation_interval = None
        self.layers = []
        self.fetched_vars = []
        self.global_training_step = 0
        self.minibatch_size = minibatch_size
        self.summary_vars = None
        self.build_network()

    """   Building network and set up training   """

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
        self.error = self.loss_function(self.target, self.output)
        self.error = tf.reduce_mean(self.error)
        tf.summary.scalar("Error", self.error)
        optimizer = self.optimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error, name="Trainer")

    """   Training network   """

    def run_training(self, sess=None, epochs=100, dir="summary"):
        session = sess if sess else tft.gen_initialized_session(dir)
        self.merge_summaries()
        self.train_network(session, self.caseman.get_training_cases(), epochs)

    def train_network(self, sess, cases, epochs=200):
        """

        Runs through the network *epochs* times with training turned on. To run it
        with training turned on it is important that the session is run with the
        training operation.

        :param sess: the session to run the training with
        :param cases: list of (feature_vector, target_vector) pairs to train the network with
        :param epochs: number of times to run through the whole case set
        :return: None

        """
        for i in range(epochs):
            step = self.global_training_step + i
            total_error = 0
            num_cases = len(cases)
            num_minibatches = math.ceil(num_cases / self.minibatch_size)

            for case_start in range(0, num_cases, self.minibatch_size):
                case_end = min(num_cases, case_start + self.minibatch_size)
                minibatch = cases[case_start:case_end]
                inputs = [c[0] for c in minibatch]
                targets = [c[1] for c in minibatch]
                feed_dict = {self.input: inputs, self.target: targets}
                _, fetched_values, _ = self.run_session(sess, [self.trainer],
                                                        fetched_vars=[self.error] + self.fetched_vars,
                                                        summary_vars=self.summary_vars,
                                                        feed_dict=feed_dict,
                                                        step=step)

                total_error += fetched_values[0]
            self.error_history.append((step, total_error / num_minibatches))
            self.maybe_run_validation(sess, step)
        self.global_training_step += epochs
        self.add_summary_plots(sess)



    def maybe_run_validation(self, sess, step):
        if self.validation_interval and (step % self.validation_interval == 0):
            cases = self.caseman.get_validation_cases()
            if len(cases) > 0:
                error = self.test_network(sess, cases, "Validation testing error: ")
                self.validation_history.append((step, error))

    """   Testing network   """

    def test_on_trains(self, sess, top_k=None):
        cases = self.caseman.get_training_cases()
        self.test_network(sess, cases, msg="Testing", top_k=top_k)

    def test_network(self, sess, cases, msg="Testing", top_k=None):
        """
        Runs through the network once with training turned off. To not do any training
        it is important to run the session without the training-operation, just the
        error operation.

        :param sess: the session to run the testing with
        :param cases: the cases to run the test on
        :param msg: a message to be printed describing the test result
        :param bestk: the number of highest ranked entries in the prediction vector to look for the target
        :return: None

        """
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        feed_dict = {self.input: inputs, self.target: targets}

        if top_k is not None:
            self.test_function = self.gen_topk_tensor(self.output, targets, top_k)
        else:
            self.test_function = self.error

        result, fetched_vars, _ = self.run_session(sess, self.test_function, self.fetched_vars, self.summary_vars,
                                                   feed_dict=feed_dict)
        if top_k is not None:
            print(msg + ": ", (100*(result/len(cases))))
        else:
            print(msg + ": ", result)
        return result

    def gen_topk_tensor(self, predictions, targets, k=1):
        is_correct = tf.nn.in_top_k(tf.cast(self.output, tf.float32), [v for v in targets],
                                    k=k)
        return tf.reduce_sum(tf.cast(is_correct, tf.int32))

    def run_session(self, session, operations, fetched_vars=None, summary_vars=None, dir="summary",
                    feed_dict=None, step=1, show_interval=0):
        """

        :param session: the session to run the operations and tensors with
        :param operations: A (list of) tensorflow operation(s) to be executed during running a session
        :param fetched_vars: Tensors to be valuated during/after running a session
        :param summary_vars: Tensors to be written to summary directory. Will be displayed in tensorboard.
        :param dir: Specifies a directory which to save the summary from running the session. Used by a FileWriter
        :param feed_dict: A tensorflow feed_dict which specify values to placeholders in the tensoflow graph
        :param step: The global training step
        :param show_interval: The frequency for displaying summary_vars
        :return: [operations values, fetched_vars values, session]
        """

        if summary_vars is not None:
            results = session.run([operations, fetched_vars, summary_vars], feed_dict=feed_dict)
            session.summary_stream.add_summary(results[2], global_step=step)
        else:
            results = session.run([operations, fetched_vars], feed_dict=feed_dict)

        if show_interval and (step % show_interval == 0):
            self.display_fetched_vars(results[1], fetched_vars, step=step)

        return results[0], results[1], session

    def merge_summaries(self):
        self.summary_vars = tf.summary.merge_all()

    def run(self, sess=None, epochs=100, top_k=None, validation_interval=None):
        self.validation_interval = validation_interval
        session = sess if sess is not None else tft.gen_initialized_session()
        self.run_training(session, epochs)
        self.test_on_trains(sess, top_k)

    def display_fetched_vars(self, fetched_vals, fetched_vars, step=1):
        names = [x.name for x in fetched_vars]
        msg = "Fetched variables at step " + str(step)
        print("\n" + msg)
        fig_index = 0
        for i, val in enumerate(fetched_vals):
            if names:
                print(">>> " + names[i] + " = ", fetched_vals[i])
            if type(val) == np.ndarray and len(val.shape) > 1:
                tft.hinton_plot(val, fig=self.fetched_var_figures[fig_index], title=names[i] + " at step " + str(step))
                fig_index += 1

            else:
                print(val, end="\n\n")

    def add_layer_summary(self, layer_index, var, spec):
        self.layers[layer_index].gen_summary(var, spec)

    def add_summary_plots(self, sess):
        error_img = draw_scatter(self.error_history, "b")
        error_image_summary = tf.summary.image('Error', error_img)

        error_vs_validation_img = draw_error_vs_validation(self.error_history, self.validation_history)
        error_vs_validation_summary = tf.summary.image("Error vs validation", error_vs_validation_img)

        summaries = tf.summary.merge([error_image_summary, error_vs_validation_summary])
        summary_vals = sess.run(summaries)
        sess.summary_stream.add_summary(summary_vals)
