import math

import tensorflow as tf
from utils.network_layer import NetworkLayer
import utils.tflowtools as tft
import numpy as np
import matplotlib.pyplot as plt
from utils.image_plotter import draw_scatter, draw_error_vs_validation, draw_hinton_plot


class Gann2:

    def __init__(self,
                 layer_sizes,
                 caseman,
                 top_k,
                 learning_rate=.1,
                 init_weight_range=(-.1, .1),
                 hidden_activation_function=tf.nn.relu,
                 output_activation_function=tf.nn.softmax,
                 loss_function=lambda labels, predictions: tf.losses.mean_squared_error(labels=labels,
                                                                                        predictions=predictions),
                 optimizer=tf.train.RMSPropOptimizer,
                 minibatch_size=10,
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
        self.accuracy_history = []
        self.validation_interval = None
        self.accuracy = None  # accuracy tensor for testing and validation
        self.case_length = None

        self.show_interval = None
        self.layers = []

        self.fetched_vars = []
        self.fetched_var_figures = []
        self.summary_vars = None

        self.global_training_step = 0
        self.minibatch_size = minibatch_size
        self.build_network()
        self.setup_training(top_k)

    """   Building network and set up training   """

    def build_network(self):
        """
        Builds the neural network in the following steps:
         - Resets the default graph
         - Initializes an input tensor to be fed with input cases during learning
         - Iterates through the layer_sizes and initializes a NetworkLayer instance with the following arguments:
            * an_network: A Gann instance to add the layer to
            * input_size: the size of the expected input to the layer
            * input_variable: the output of the layer before it
            * output_size: the size of the next layer
            * index: the layer index (0 = first hidden layer, etc.)
            * init_weight_range: the initial weight range for the random computed weights for the layer
            * activation_function: the activation function for the layer
         - Sets the output variable (output of whole network) to be the aoutput of the last layer
         - If an output activation function is given, sets the output variable to this
         - Initializes a target tensor to be fed with prefered target cases/labels for the training

        :return:
        """

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

    def add_layer(self, layer):
        """

        Adds a layer to the network

        :param layer: a layer of type NetworkLayer
        :return: None

        """
        self.layers.append(layer)

    def setup_training(self, top_k=1):
        """

        Makes the network ready to be trained. Is called before 'run_training()'
         - Sets the error variable to be the average value of the loss_function provided from the user
         - Sets a trainer to be an optimizer provided from the user
         - Sets a case_length tensor placeholder to be used when computing the accuracy
         - Sets an accuracy tensor to be the top_k

        :param sess: session which to run the training with
        :param top_k: top_k value
        :return: None

        """
        self.error = self.loss_function(self.target, self.output)
        self.error = tf.reduce_mean(self.error, name="Error")
        optimizer = self.optimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error, name="Trainer")

        self.case_length = tf.placeholder(tf.int32, shape=None, name="Case_length")
        topk_tensor = self.gen_topk_tensor(self.output, self.target, top_k)
        self.accuracy = tf.multiply(
            tf.constant(100, dtype=tf.float64),
            tf.divide(
                tf.reduce_sum(tf.cast(topk_tensor, tf.int32)),
                self.case_length),
            name="Accuracy")

    """   Training network   """

    def run_training(self, sess=None, epochs=100, dir="summary"):
        """

        Runs the training session

        :param sess: the session which to run the training with
        :param epochs: the number of epochs (number of runs through the whole data set)
        :param dir: path to the directory which to store the session summaries
        :return: None
        """
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
            step = self.global_training_step + i + 1
            total_error = 0
            num_cases = len(cases)
            num_minibatches = math.ceil(num_cases / self.minibatch_size)

            for case_start in range(0, num_cases, self.minibatch_size):
                case_end = min(num_cases, case_start + self.minibatch_size)
                minibatch = cases[case_start:case_end]
                inputs = [c[0] for c in minibatch]
                targets = [c[1] for c in minibatch]
                feed_dict = {
                    self.input: inputs,
                    self.target: targets,
                    self.case_length: len(minibatch)
                }
                _, fetched_values, _ = self.run_session(sess,
                                                        [self.trainer],
                                                        fetched_vars=[self.error] + self.fetched_vars,
                                                        summary_vars=self.summary_vars,
                                                        feed_dict=feed_dict,
                                                        step=step)

                total_error += fetched_values[0]
            self.error_history.append((step, total_error / num_minibatches))
            self.maybe_run_validation(sess, step)

            if self.show_interval and (step % self.show_interval == 0):
                self.show_fetched_vars(sess, fetched_values[1:], self.fetched_vars, step=step)
        self.global_training_step += epochs
        self.add_summary_plots(sess)

    def maybe_run_validation(self, sess, step):
        if self.validation_interval and (step % self.validation_interval == 0):
            cases = self.caseman.get_validation_cases()
            if len(cases) > 0:
                self.test_network(sess, cases, "Validation, step %d" % step, step=step)

    """   Testing network   """

    def test_on_trains(self, sess):
        cases = self.caseman.get_training_cases()
        self.test_network(sess, cases, msg="Testing on training set")

    def test_network(self, sess, cases, msg="Testing", step=None):
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
        feed_dict = {
            self.input: inputs,
            self.target: targets,
            self.case_length: len(cases)
        }

        operations = [self.error, self.accuracy]
        operation_names = ["Error", "Accuracy"]

        results, _, _ = self.run_session(sess, operations, self.fetched_vars,
                                         feed_dict=feed_dict)
        print(msg + ":")

        for res, name in zip(results, operation_names):
            print("  >>> %s: %f" % (name, res))
        return results

    def gen_topk_tensor(self, prediction, target, k=1):
        prediction = tf.cast(prediction, tf.float32)

        target = tf.argmax(tf.cast(target, tf.int64), 1)

        topk_tensor = tf.nn.in_top_k(predictions=prediction, targets=target, k=k)
        return topk_tensor

    def run_session(self, session, operations, fetched_vars=None, summary_vars=None, dir="summary",
                    feed_dict=None, step=1):
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

        return results[0], results[1], session

    def merge_summaries(self):
        self.summary_vars = tf.summary.merge_all()

    def run(self, sess=None, epochs=100, validation_interval=None, show_interval=None):
        plt.ion()
        self.validation_interval = validation_interval
        self.show_interval = show_interval
        session = sess if sess is not None else tft.gen_initialized_session()
        session.run(tf.global_variables_initializer())
        self.run_training(session, epochs)
        self.test_on_trains(sess)
        plt.ioff()

    def show_fetched_vars(self, sess, fetched_vals, fetched_vars, step=1):
        for i, val in enumerate(fetched_vals):
            if type(val) == list:
                fig = draw_hinton_plot(val, headline="Hinton plot")
                sum = tf.summary.image("Error vs validation", fig)
                summary_vals = sess.run(sum)
                sess.summary_stream.add_summary(summary_vals)
            else:
                msg = "Fetched variables at step " + str(step)
                print("\n" + msg)
                print("  >>> " + fetched_vars[i].name + " = ", val, end="\n\n")

    def add_summary(self, *args, **kwds):
        if isinstance(args[0], int):
            self.add_summary_from_layer(*args, **kwds)
        else:
            self.add_summary_from_tensor(*args, **kwds)

    def add_summary_from_tensor(self, var: tf.nn, type="scalar"):
        if type == "scalar":
            tf.summary.scalar(var.name, var)
        elif type == "hist":
            tf.summary.histogram(var.name, var)
        elif type == "image":
            tf.summary.image(var.name, var)

    def add_summary_from_layer(self, layer_index: int, var: str, spec):
        self.layers[layer_index].gen_summary(var, spec)

    def add_fetched_var(self, layer_index, val_type="weight"):
        if type(val_type) == list:
            self.fetched_vars.append([self.layers[layer_index].get_var(val_type[i]) for i in range(len(val_type))])
        else:
            self.fetched_vars.append(self.layers[layer_index].get_var(val_type))

    def add_summary_plots(self, sess):
        summaries = []
        if len(self.error_history) > 0:
            error_img = draw_scatter(self.error_history, "r")
            summaries.append(tf.summary.image('Error', error_img))

            if len(self.validation_history) > 0:
                error_vs_validation_img = draw_error_vs_validation(self.error_history, self.validation_history)
                summaries.append(tf.summary.image("Error vs validation", error_vs_validation_img))

        if len(self.accuracy_history) > 0:
            accuracy_img = draw_scatter(self.accuracy_history, "b")
            summaries.append(tf.summary.image("Validation accuracy", accuracy_img))

        summaries = tf.summary.merge(summaries)
        summary_vals = sess.run(summaries)
        sess.summary_stream.add_summary(summary_vals)
