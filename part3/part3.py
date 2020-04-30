"""
array => [5,6,2,1]
shape => (4,)
type => 1D array, Vector

-------

array => [[1,5,6,2],[3,2,1,3]]
shape => (2,4)
type => 2D Array, Matrix

-------

array[[[1,5,6,2],[3,2,1,3]],[[5,2,1,2],[6,4,8,4]],[[2,8,5,3],[1,1,9,4]]]
shape => (3,2,4)
type => 3D Array




what is a tensor ?
*a tensor is an object that can be
represented as an array.

"""



import numpy as np

#how do we multiply inputs and weights ? ==> dot product

inputs = [1.0,2.0,3.0,2.5]

weights = [0.2,0.8,-0.5,1.0]

bias = 2

# dot product function => order of parameters can be important.
output = np.dot(weights,inputs) + bias
print (output)



new_weights =   [[0.2,0.8,-0.5,1.0],
            [0.5,-0.91,0.26,-0.5],
            [-0.26,-0.27,0.17,0.87]]

new_biases = [2,3,0.5]

output = np.dot(new_weights,inputs) + new_biases
print (output)
