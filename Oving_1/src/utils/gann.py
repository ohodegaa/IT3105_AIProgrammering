import tensorflow as tf
import numpy as np
import math
from utils import tflowtools as tft
import matplotlib.pyplot as plt

from utils.gann_module import Gannmodule


class Gann:

    def __init__(self,
                 dims,
                 cman,
                 lrate=.1,
                 showint=None,
                 mbs=10,
                 vint=None,
                 softmax=False,
                 hidden_act_func=tf.nn.relu,
                 output_act_func=None,
                 loss_function=None,
                 init_weight_range=(-.1, .1),
                 optimizer=tf.train.AdamOptimizer):
        self.learning_rate = lrate
        self.layer_sizes = dims  # Sizes of each layer of neurons
        self.show_interval = showint  # Frequency of showing grabbed variables
        self.global_training_step = 0  # Enables coherent data-storage during extra training runs (see runmore).
        self.grabvars = []  # Variables to be monitored (by gann code) during a run.
        self.grabvar_figures = []  # One matplotlib figure for each grabvar
        self.minibatch_size = mbs
        self.validation_interval = vint
        self.validation_history = []
        self.caseman = cman
        self.softmax_outputs = softmax
        self.hidden_act_func = hidden_act_func
        self.output_act_func = output_act_func
        self.loss_function = loss_function
        self.init_weight_range = init_weight_range
        self.modules = []
        self.optimizer = optimizer
        self.build()

    # Probed variables are to be displayed in the Tensorboard.
    def gen_probe(self, module_index, type, spec):
        self.modules[module_index].gen_probe(type, spec)

    # Grabvars are displayed by my own code, so I have more control over the display format.  Each
    # grabvar gets its own matplotlib figure in which to display its value.
    def add_grabvar(self, module_index, type='wgt'):
        self.grabvars.append(self.modules[module_index].getvar(type))
        self.grabvar_figures.append(plt.figure())

    def roundup_probes(self):
        self.probes = tf.summary.merge_all()

    def add_module(self, module):
        self.modules.append(module)

    def build(self):
        tf.reset_default_graph()  # This is essential for doing multiple runs!!
        num_inputs = self.layer_sizes[0]  # number of input nodes
        self.input = tf.placeholder(tf.float64, shape=(None, num_inputs), name='Input')  # input layer
        invar = self.input
        insize = num_inputs

        # Build all of the modules
        for i, outsize in enumerate(self.layer_sizes[1:]):
            # for all the hidden layers, create a Gannmodule (hidden layer)
            gmod = Gannmodule(self, i, invar, insize, outsize, activation_function=self.hidden_act_func,
                              init_weight_range=self.init_weight_range)
            invar = gmod.output
            insize = gmod.outsize

        self.output = gmod.output  # Output of last module is output of whole network

        if self.output_act_func is not None:
            self.output = self.output_act_func(self.output)

        elif self.softmax_outputs:  # softmax kind of normalizes the output, so they sum to 1
            self.output = tf.nn.softmax(self.output)
        # if both are given the given activation function will be used

        self.target = tf.placeholder(tf.float64, shape=(None, gmod.outsize), name='Target')
        self.configure_learning()

    # The optimizer knows to gather up all "trainable" variables in the function graph and compute
    # derivatives of the error function with respect to each component of each variable, i.e. each weight
    # of the weight array.

    def configure_learning(self):
        if type(self.loss_function) == "str":
            self.error = {
                "mse": lambda: tf.reduce_mean(tf.square(self.target - self.output)),
                "scel": lambda: tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target, logits=self.output)),
                "smce": lambda: tf.losses.softmax_cross_entropy(onehot_labels=self.target, logits=self.output,
                                                                reduction=tf.losses.Reduction.MEAN),
            }[self.loss_function]()
        elif type(self.loss_function) == "function":
            self.error = self.loss_function(self.target, self.output)
        else:
            self.error = tf.reduce_mean(tf.square(self.target - self.output))
        self.predictor = self.output  # Simple prediction runs will request the value of output neurons
        # Defining the training operator
        optimizer = self.optimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error, name='Backprop')

    def do_training(self, sess, cases, epochs=100, continued=False):
        if not (continued): self.error_history = []
        for i in range(epochs):
            error = 0
            step = self.global_training_step + i
            gvars = [self.error] + self.grabvars
            mbs = self.minibatch_size
            ncases = len(cases)
            nmb = math.ceil(ncases / mbs)
            for cstart in range(0, ncases, mbs):  # Loop through cases, one minibatch at a time.
                cend = min(ncases, cstart + mbs)
                minibatch = cases[cstart:cend]
                inputs = [c[0] for c in minibatch]
                targets = [c[1] for c in minibatch]
                feeder = {self.input: inputs, self.target: targets}
                _, grabvals, _ = self.run_one_step([self.trainer], gvars, self.probes, session=sess,
                                                   feed_dict=feeder, step=step, show_interval=self.show_interval)
                error += grabvals[0]
            self.error_history.append((step, error / nmb))
            self.consider_validation_testing(step, sess)
        self.global_training_step += epochs
        tft.plot_training_history(self.error_history, self.validation_history, xtitle="Epoch", ytitle="Error",
                                  title="", fig=not (continued))

    # bestk = 1 when you're doing a classification task and the targets are one-hot vectors.  This will invoke the
    # gen_match_counter error function. Otherwise, when
    # bestk=None, the standard MSE error function is used for testing.

    def do_testing(self, sess, cases, msg='Testing', bestk=None):
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        self.test_func = self.error
        if bestk is not None:
            self.test_func = self.gen_match_counter(self.predictor, [tft.one_hot_to_int(list(v)) for v in targets],
                                                    k=bestk)
        testres, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, self.probes, session=sess,
                                                 feed_dict=feeder, show_interval=None)
        if bestk is None:
            print('%s Set Error = %f ' % (msg, testres))
        else:
            print('%s Set Correct Classifications = %f %%' % (msg, 100 * (testres / len(cases))))
        return testres  # self.error uses MSE, so this is a per-case value when bestk=None

    # Logits = tensor, float - [batch_size, NUM_CLASSES].
    # labels: Labels tensor, int32 - [batch_size], with values in range [0, NUM_CLASSES).
    # in_top_k checks whether correct val is in the top k logit outputs.  It returns a vector of shape [batch_size]
    # This returns a OPERATION object that still needs to be RUN to get a count.
    # tf.nn.top_k differs from tf.nn.in_top_k in the way they handle ties.  The former takes the lowest index, while
    # the latter includes them ALL in the "top_k", even if that means having more than k "winners".  This causes
    # problems when ALL outputs are the same value, such as 0, since in_top_k would then signal a match for any
    # target.  Unfortunately, top_k requires a different set of arguments...and is harder to use.

    def gen_match_counter(self, logits, labels, k=1):
        correct = tf.nn.in_top_k(tf.cast(logits, tf.float32), labels, k)  # Return number of correct outputs
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def training_session(self, epochs, sess=None, dir="probeview", continued=False):
        self.roundup_probes()
        session = sess if sess else tft.gen_initialized_session(dir=dir)
        self.current_session = session
        self.do_training(session, self.caseman.get_training_cases(), epochs, continued=continued)

    def testing_session(self, sess, bestk=None):
        cases = self.caseman.get_testing_cases()
        if len(cases) > 0:
            self.do_testing(sess, cases, msg='Final Testing', bestk=bestk)

    def consider_validation_testing(self, epoch, sess):
        # What's happening here?

        if self.validation_interval and (epoch % self.validation_interval == 0):
            cases = self.caseman.get_validation_cases()
            if len(cases) > 0:
                error = self.do_testing(sess, cases, msg='Validation Testing')
                self.validation_history.append((epoch, error))

    # Do testing (i.e. calc error without learning) on the training set.
    def test_on_trains(self, sess, bestk=None):
        self.do_testing(sess, self.caseman.get_training_cases(), msg='Total Training', bestk=bestk)

    # Similar to the "quickrun" functions used earlier.

    def run_one_step(self, operators, grabbed_vars=None, probed_vars=None, dir='probeview',
                     session=None, feed_dict=None, step=1, show_interval=1):
        sess = session if session else tft.gen_initialized_session(dir=dir)
        if probed_vars is not None:
            results = sess.run([operators, grabbed_vars, probed_vars], feed_dict=feed_dict)
            sess.probe_stream.add_summary(results[2], global_step=step)
        else:
            results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
        if show_interval and (step % show_interval == 0):
            self.display_grabvars(results[1], grabbed_vars, step=step)
        return results[0], results[1], sess

    def display_grabvars(self, grabbed_vals, grabbed_vars, step=1):
        names = [x.name for x in grabbed_vars]
        msg = "Grabbed Variables at Step " + str(step)
        print("\n" + msg, end="\n")
        fig_index = 0
        for i, v in enumerate(grabbed_vals):
            if names: print("   " + names[i] + " = ", end="\n")
            if type(v) == np.ndarray and len(v.shape) > 1:  # If v is a matrix, use hinton plotting
                tft.hinton_plot(v, fig=self.grabvar_figures[fig_index], title=names[i] + ' at step ' + str(step))
                fig_index += 1
            else:
                print(v, end="\n\n")

    def run(self, epochs=100, sess=None, continued=False, bestk=None):
        plt.ion()
        self.training_session(epochs, sess=sess, continued=continued)
        self.test_on_trains(sess=self.current_session, bestk=bestk)
        self.testing_session(sess=self.current_session, bestk=bestk)
        self.close_current_session(view=False)
        plt.ioff()

    # After a run is complete, runmore allows us to do additional training on the network, picking up where we
    # left off after the last call to run (or runmore).  Use of the "continued" parameter (along with
    # global_training_step) allows easy updating of the error graph to account for the additional run(s).

    def runmore(self, epochs=100, bestk=None):
        self.reopen_current_session()
        self.run(epochs, sess=self.current_session, continued=True, bestk=bestk)

    #   ******* Saving GANN Parameters (weights and biases) *******************
    # This is useful when you want to use "runmore" to do additional training on a network.
    # spath should have at least one directory (e.g. netsaver), which you will need to create ahead of time.
    # This is also useful for situations where you want to first train the network, then save its parameters
    # (i.e. weights and biases), and then run the trained network on a set of test cases where you may choose to
    # monitor the network's activity (via grabvars, probes, etc) in a different way than you monitored during
    # training.

    def save_session_params(self, spath='netsaver/my_saved_session', sess=None, step=0):
        session = sess if sess else self.current_session
        state_vars = []
        for m in self.modules:
            vars = [m.getvar('wgt'), m.getvar('bias')]
            state_vars = state_vars + vars
        self.state_saver = tf.train.Saver(state_vars)
        self.saved_state_path = self.state_saver.save(session, spath, global_step=step)

    def reopen_current_session(self):
        self.current_session = tft.copy_session(self.current_session)  # Open a new session with same tensorboard stuff
        self.current_session.run(tf.global_variables_initializer())
        self.restore_session_params()  # Reload old weights and biases to continued from where we last left off

    def restore_session_params(self, path=None, sess=None):
        spath = path if path else self.saved_state_path
        session = sess if sess else self.current_session
        self.state_saver.restore(session, spath)

    def close_current_session(self, view=True):
        self.save_session_params(sess=self.current_session)
        tft.close_session(self.current_session, view=view)
