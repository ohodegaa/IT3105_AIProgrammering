# Module 1

## Required Parameters

### Network dimension:
A list of layer sizes. Each element specify the number
of neurons in the corresponding layer.
 
- The first elements specify 
the number of input neurons in the network and should
be equal to the size of case vectors in the data set.
- The subsequent (except the last) elements can be
of arbitrary sizes and specify the sizes (number of 
neurons) of the hidden layers.
- The last element specify the size (number of neurons)
of the output (last hidden layer) and should be equal
to the number of elements in the target vectors in the
data set 

##### Example: 
```python
import random
data_set = [
    [[2, 4, 5, 7], [1, 0]],
    [[9, 1, 3, 7], [0, 1]],
]
input_size = len(data_set[0][0])
output_size = len(data_set[0][1])
dims = [input_size, random.randint(1, 10), random.randint(1, 10), output_size]
```


##### Where to set:

```python
Gann(dims, ...)
```


### Hidden activation function:
##### Example: 
```python
tf.nn.relu
tf.nn.sigmoid
tf.nn.leaky_relu
tf.nn.crelu
...
```


##### Where to set:

```python
Gann(..., hidden_act_func)
```

### Output activation function:
##### Example: 
```python
tf.nn.softmax
tf.nn.relu
tf.nn.sigmoid
tf.nn.leaky_relu
tf.nn.crelu
...
```


##### Where to set:

```python
Gann(..., output_act_func)
```


### Cost function
*Also called a loss function*
##### Example: 
```python
"mse" # mean squared error
"sice" # sigmoid cross entropy
"soce" # softmax cross entropy
...
```


##### Where to set:

```python
Gann(..., cost_func)
```

##### default:
```python
tf.losses.mean_squared_error
```

### Learning rate

##### Example:
```python
0.1
0.01
```

##### Where to set:

```python
Gann(..., lrate)
```

##### Default:
```python
0.1
```

### Initial weight range:

##### Example:
```python
(-.1, .1)
```

##### Where to set:
```python
Gann(..., init_weight_range)
```

##### Default:
```python
(-.1, .1)
```


### Optimizer:

##### Example:
```python
import tensorflow as tf
tf.train.GradientDescentOptimizer
tf.train.RMSPropOptimizer
tf.train.AdagradOptimizer
tf.train.AdamOptimizer
```

##### Where to set:
```python
Gann(..., optimizer)
```

##### Default:
```python
import tensorflow as tf
tf.train.GradientDescentOptimizer
```


### Data source
*The cfunc must be a callable function with no arguments, which returns a list of cases of the form (input_target, target)*

##### Where to set
```python
Caseman(cfunc, ...)
```

##### Example:
```python
import utils.tflowtools as tft

# functions
cfunc = lambda: tft.gen_all_parity_cases(num_bits=8)
cfunc = lambda: tft.gen_dense_autoencoder_cases(count=10, size=5)
...

# data files
cfunc = lambda: utils.readFile("<file_name>")
```


### Case fraction:

##### Where to set:
```python
Caseman(..., cfrac)
```

##### Exmaple:
```python
0.8
0.9
```

##### Default:
```python
1.0 # the whole data set
```


### Validation fraction:

##### Where to set:
```python
Caseman(..., vfrac)
```

##### Exmaple:
```python
0.1
0.2
```

##### Default:
```python
0
```

### Validation interval:

##### 
