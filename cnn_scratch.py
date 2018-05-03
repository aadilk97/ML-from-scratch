import numpy as np
import math

def convolution(x, W, s, nf, f, n):
	## Performs the convolution operation

	op_size = math.floor((n-f) / s + 1)
	op = np.zeros((1, nf, op_size, op_size))

	for i in range(nf):
		for h in range(op_size):
			for w in range(op_size):
				op[0, i, h, w] = np.sum(x[0, :, h*s:h*s+f, w*s:w*s+f] * W[i, :, :, :]) 

	return op


def sigmoid(X):
 
    X = 1/(1+np.exp(-X))
    return X




## Initializing input of shape(32, 32) 
## Initialization format --> (n_samples, n_channels, height, width)
x = np.random.uniform(-1, 1,  size=(1, 1, 32, 32))

## Initializing output of shape (10, 1)
y = np.random.uniform(0, 1, size=(10, 1))


## Initializing the filters and the bias for the filters
w1 = np.random.uniform(-1, 1, size=(4, 1, 3, 3))
b1 = np.random.uniform(-1, 1, size=(1, 4, 1, 1))

## Intializing the number of filters, frame size, strides, output size.
nf = w1.shape[0]
f = w1.shape[2]
n = x.shape[2]
s = 2
op_size = math.floor((n-f) / s + 1)

## Initializing w2 and b2
w2 = np.random.uniform(-1, 1, size=(1024, nf * op_size * op_size))
b2 = np.random.uniform(-1, 1, size=(1024))

## Initializing w3 and b3
w3 = np.random.uniform(-1, 1, size=(10, 1024))
b3 = np.random.uniform(-1, 1, size=(10))


## Sample training for 2 passes (Forward and backward pass)
num_epochs = 2
for epoch in range(num_epochs):
	## ------------------------- Start of forward propagation ------------------------------------

	## Performing covolution of the input with the filters
	## Shape (1, 4, 15, 15)
	z1 = convolution(x, w1, s, nf, f, n) + b1

	## Applying relu activation(non-linearity) to the output of convolution
	z1 = np.maximum(z1, 0)

	## Flattening out the output of convolution 
	## Shape (1, 900)
	z2 = z1.reshape((1, z1.shape[1] * z1.shape[2] * z1.shape[3]))


	## Fully connected layer
	## Shape (1, 1024)
	fully_connected = np.dot(z2, w2.T) + b2

	## Applying sigmoid activation to the fully-connected layer to add non linearity to the layer.
	fully_connected = sigmoid(fully_connected)


	## Final output layer
	## Shape (1, 10) -- 10 output classes
	output = np.dot(fully_connected, w3.T) + b3
	output = output.reshape(10, 1)
	output = sigmoid(output)


	## ------------------------- End of forward propagation ------------------------------------

	


	## ------------------------- Start of back propagation ------------------------------------

	## Compututing the delta terms and the derivative of the weights wrt the error for each layer
	dz4 = output - y

	dz3 = np.dot(w3.T, dz4).reshape(1024, 1) * (fully_connected * (1 - fully_connected)).reshape(1024,1)
	delta3 = np.dot(dz4, fully_connected)
	db3 = np.sum(dz4, axis=1)

	dz2 = np.dot(w2.T, dz3).reshape(nf * op_size * op_size, 1) * (z2 * (1 - z2)).reshape(nf * op_size * op_size, 1)
	delta2 = np.dot(dz3, z2)
	db2 = np.sum(dz3, axis=1)


	dz2 = dz2.reshape(1, 4, 225)
	db1 = np.zeros((4,1))

	for i in range(nf):
		db1[i] = np.sum(dz2[0, i])

	dz2 = dz2.reshape(1, 4, 15, 15)
	db1 = db1.reshape(1, 4, 1, 1)

	delta1 = np.zeros(w1.shape)
	for i in range(nf):
		for h in range(op_size):
			for w in range(op_size):
				delta1[i] += x[0, :, h*s:h*s+f, w*s:w*s+f] * dz2[0, i, h, w]


	## Updating the weights and biases 			
	learning_rate = 0.01 
	w3 -= learning_rate * delta3
	w2 -= learning_rate * delta2
	w1 -= learning_rate * delta1

	b3 -= learning_rate * db3
	b2 -= learning_rate * db2
	b1 -= learning_rate * db1

	## ------------------------- End of back propagation ------------------------------------


