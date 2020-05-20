"""
Why are we using batch ?

** The primary reason, we're calculating things in parallel the bigger the batch
the more parallel operations that we can run now this is also why we tend to do
neural network training on GPUs rather than doing them on CPUs so I don't know
what the average CPU core these days some are probably between 4 and 8
that's not very many course compared to your typical GPU which is
going to have hundreds or thousands of these cores that we can actually run
calculations on so that's why it's so much faster on a GPU.

** The other reason why we do things in batches is because it helps with
generalization. If you give all data one by one it will be so slow and every
time it needs to fit data again and again. So if we make it batch = 4, it will
give data four by four so in the end it will be faster. But we can't give all
data at the same time, it's over fitting.

generally = 32,64,128

we will learn how to use batch, giving many inputs at the same time.
"""

"""
import numpy as np

# batch = 4
inputs = [[1.0,2.0,3.0,2.5],
            [2.0,5.0,-1.0,2.0],
                [-1.5,2.7,3.3,-0.8]]

weights =   [[0.2,0.8,-0.5,1.0],
            [0.5,-0.91,0.26,-0.5],
            [-0.26,-0.27,0.17,0.87]]

biases = [2,3,0.5]

# there is no mathematical way for input * weights (matrix multiplication)
# so we need to use transpose

weights2 =   [[0.1,-0.14, 0.5],
            [-0.5,0.12,-0.33],
            [-0.44,0.73,-0.13]]

biases2 = [-1,2,-0.5]


layer1 = np.dot(inputs, np.array(weights).T) + biases
layer2 = np.dot(layer1, np.array(weights2).T) + biases2

print(layer2)
"""

############################
############################
############################


"""
How do we initialize our neural network ?
1) We need to initialize our weights : we want small values like -1 and 1
    because we want to scale our inputs but if weights are bigger than 1 or -1
    output number will increase and increase and increase, in the end it can be
    explosion. In this case, we can just start to use code between -0.1 and 0.1

2) For biases, we will initialize with zero values. Sometimes it's not good
    to do that because whenever inputs*weights give the value like zero for some
    reason, we will add zero value from bias and output will be zero. After that
    other multiplications will be zero, because of this output. (network is dead.)

"""
"""
import numpy as np

np.random.seed(0)

# batch = 4
X = [[1.0,2.0,3.0,2.5],[2.0,5.0,-1.0,2.0],[-1.5,2.7,3.3,-0.8]]
# X is kind of a standart ML input feature sets.


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4,3)
#layer2 = Layer_Dense(3,4) # n_inputs = previous layer's n_neurons

layer1.forward(X)
layer2.forward(layer1.output)
print (layer1.output)
