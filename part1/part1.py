"""

*purpose is learning neural networks at a deep level.

*every neuron has unique weight and every neuron has unique bias.

example :

    Layer sizes : 10, 128, 128, 128, 2

    biases = 10+128+128+128+2 = 396
    weights = (10x128)+(128x128)+(128x128)+(128x2) = 34304


*the hard parf of neural networks and deep learning is figuring out
how to tune things.


"""






"""
*In this fully connected neural network, every neuron has unique connection
to every single previous neuron.

*let's say there are 3 neurons
that are feeding into this neuron that we are gonna build.

"""

inputs = [1.2,5.1,2.1]

#every unique input is also going to have unique weight to associated with it.

weights = [3.1,2.1,8.7]

#every unique neuron has a unique bias.

bias = 3

output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias
print (output)
