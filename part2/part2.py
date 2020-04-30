"""

* an example
that you're trying to predict failure or
not failure for servers or computer
equipment right and so you've got
various sensors like heat sensors and
humidity sensors etc.

* each value here in these inputs
if this is your input layer each unique
value here could be a value from a
separate sensor.


what if we want to model three neurons with
four inputs ?

"""

inputs = [1.0,2.0,3.0,2.5]
weights1 = [0.2,0.8,-0.5,1.0]
weights2 = [0.5,-0.91,0.26,-0.5]
weights3 = [-0.26,-0.27,0.17,0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5


output =   [inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2] + inputs[3] * weights1[3] + bias1,
            inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2] + inputs[3] * weights2[3]+ bias2,
            inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2] + inputs[3] * weights3[3]+ bias3]


print (output)


"""
* inputs you can't really
change the inputs because the inputs are
outputs from like either a previous
layer or they're like your actual data
from your actual input data
from features like sensor data or
something like that.


* you really can
never change the inputs directly but you
actually can by changing weights and
biases so you can change because the
inputs.


"""


#we can simplify our code and make much more dynamic

inputs = [1.0,2.0,3.0,2.5]

weights =   [[0.2,0.8,-0.5,1.0],
            [0.5,-0.91,0.26,-0.5],
            [-0.26,-0.27,0.17,0.87]]

biases = [2,3,0.5]

layer_outputs = []

for neuron_weights, neuron_bias in zip(weights,biases): #combine together
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)
