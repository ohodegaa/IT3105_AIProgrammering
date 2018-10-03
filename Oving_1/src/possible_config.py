import tensorflow as tf

"""
Many of the loss function needs different named parameters, but they often refer
to the same, namely the target and the output. By providing it as an array
function to the gann we can easily control any additional parameters and
make the gann highly general in case of which loss function to use
"""


loss_functions = {
    lambda labels, predictions: tf.losses.mean_squared_error(labels=labels, predictions=predictions),
    lambda labels, predictions: tf.losses.absolute_difference(labels=labels, predictions=predictions),
    lambda labels, predictions: -tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=predictions),
    lambda labels, predictions: tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=predictions),
    lambda labels, predictions: tf.losses.hinge_loss(labels=labels, logits=predictions),
    lambda labels, predictions: tf.losses.huber_loss(labels=labels, predictions=predictions),
    lambda labels, predictions: tf.losses.mean_pairwise_squared_error(labels=labels, predictions=predictions),
    lambda labels, predictions: tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=predictions),
    lambda labels, predictions: tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=predictions)
}

output_activation_functions = [
    tf.nn.relu,
    tf.nn.relu6,
    tf.nn.elu,
    tf.nn.selu,
    tf.nn.softplus,
    tf.nn.softsign,
    tf.nn.sigmoid,
    tf.nn.tanh,
]

hidden_activation_functions = [
    tf.nn.relu,
    tf.nn.relu6,
    tf.nn.elu,
    tf.nn.selu,
    tf.nn.softplus,
    tf.nn.softsign,
    tf.nn.sigmoid,
    tf.nn.tanh,
]

optimizers = [
    tf.train.RMSPropOptimizer,
    tf.train.GradientDescentOptimizer,
    tf.train.AdagradOptimizer,
    tf.train.AdadeltaOptimizer,
    lambda lrate: tf.train.MomentumOptimizer(learning_rate=lrate, momentum=0.5),
    tf.train.AdamOptimizer,
    tf.train.FtrlOptimizer,
]

case_sets = {

}