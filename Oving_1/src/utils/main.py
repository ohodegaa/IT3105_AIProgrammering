import tensorflow as tf
import utils.tflowtools as tft

class Config_network:

    def __init__(self):

        self.network_dim = [8,4,2]          # The number og layers in the network along with the size of each layer
        self.hidden_activation_function = tf.nn.leaky_relu      # Function used for all hidden layers
        self.output_activation_function = tf.nn.softmax_cross_entropy_with_logits  # Function for the last layer (output)
        self.cost_function = tf.losses.mean_squared_error   # Loss function, such as mean-squared error
        self.learning_rate = 0.1    # Learning rate for the weights and biases
        self.initial_weight_range = (-.1, .1)   # An upper and an lower bound, ot be used when random initializing all weights in the network
        self.optimizer = tf.train.GradientDescentOptimizer      # Choose between: Gradient descent, RMSprop, Adagrad and ADAM
        self.data_source = None         # Data Source - specified as either: a data file or a function name (such as one of those in tflowtools.py)
        self.case_fraction = 1.0        # Choose how much of the data source you want to use
        self.validation_fraction = 0.1  # The fraction of data cases to be used for validation testing
        self.validation_interval = None # Number of training minibatches between each validation test
        self.test_fraction = 0.1        # The fraction of data cases to be used for standard testing
        self.minibatch_size = 1         # The number of training cases in a minibatch
        self.map_batch_size = 0         # The number og training cases to be used for a map test. A value of 0 indicates that no map test will be performed
        self.steps = 5                  # The total number of minibatches to be run through the system during traning
        self.map_layers = 1             # The layer to be visualized during the map test
        self.map_dendrograms = []       # List og layers whose activation patterns (during the map test) will be used to produce dendrograms, one per specified layer
        self.display_weight = []        # List of the weight arrays to be visualized at the end of the run
        self.display_biases = []        # List of the bias vectors to be visualized at the end of the run

    # Parity
    def parity_cases(self):
        num_bits = input(" Number of bits: ")      # Number of bits in each case
        double = input("Double: ")     # When double=True, a 2-bit vector is the target, with bit 0 indicating even parity and bit 1 indicating odd parity.
        return tft.gen_all_parity_cases(num_bits=num_bits, double=double)

    # Symmerty
    def symmetry_cases(self):
        length_vector = input("Length of the bit vector: ")
        count = input("Count: ")
        lable = 1
        return tft.gen_symvect_cases(length_vector,count,lable)

    # Autoencoder
    def all_one_hot_cases(self):
        length = input("Length: ")
        floats = False
        return tft.gen_all_one_hot_cases(length,floats)

    # Autoencoder
    def dense_autoencoder_cases(self):
        count = input("Count: ")
        size = input("Size: ")
        density_range = (0,1)
        return tft.gen_dense_autoencoder_cases(count, size, density_range)

    # Bit Counter
    def vector_count_cases(self):
        number = input("Number: ")
        size = input("Size: ")
        density_range = (0, 1)
        random = True
        pop_target = True
        return tft.gen_vector_count_cases(number, size, density_range, random, pop_target)

    # Segment Counter
    def segmented_vector_cases(self):
        vector_length = input("Vector length: ")
        number_of_segments = input("Number of segments: ")
        min_segment = input("Minimum number og segments in a vector: ")
        max_segment = input("Maximum number of segments in a vector: ")
        pop_target = True
        return tft.gen_segmented_vector_cases(vector_length,number_of_segments,min_segment,max_segment,pop_target)




    def main(self):
        scenario = input()
        cases = {
            "parity":
            " symmetry":
        }