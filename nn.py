## Neural network for solving xor

import numpy as np
import math

def sigmoid(X):
 
    X = 1/(1+np.exp(-X))
    return X

def sigmoid2(X):
	return (1 / (1 + math.exp(-X)))

def round(X):
	if X >= 0.5:
		return 1

	else: 
		return 0

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

X = X.T

num_inputs = X.shape[0]
l2_nodes = 2

input_layer = np.ones(num_inputs)
W1 = np.random.randn(l2_nodes, num_inputs)
b1 = np.random.randn(l2_nodes)
 
W2 = np.random.randn(1, l2_nodes)
b2 = np.random.randn(1,1)



learning_rate = 0.05
for epoch in range(100000):
	delta1 = 0
	delta2 = 0
	db1 = db2 = 0
	total_error = 0
	for i in range(X.shape[1]):
		input_layer = X[:, i]

		z1 = np.matmul(W1, input_layer) + b1
		a1 = sigmoid(z1)


		z2 = np.matmul(W2, a1) + b2
		a2 = sigmoid(z2)

		error = 0.5 * math.pow((a2 - y[i]), 2)

		dz2 = a2 - y[i]
		db2 += dz2
		dz1 = np.dot(W2.T, dz2) * (a1 * (1 - a1)).reshape(l2_nodes, 1)
		db1 += np.sum(dz1, axis=1)

		delta2 += dz2 * a1.T
		input_layer = input_layer.reshape(X.shape[0], 1)
		delta1 += np.matmul(dz1, input_layer.T)

		total_error += error

	delta1 = delta1 / X.shape[1]
	delta2 = delta2 / X.shape[1]
	db1 /= X.shape[1]
	db2 /= X.shape[1]

	W1 -= learning_rate * delta1
	W2 -= learning_rate * delta2
	b2 -= learning_rate * db2
	b1 -= learning_rate * db1


	print ("Error = ", total_error)


for i in range(X.shape[1]):
	input_layer = X[:, i]

	z2 = np.matmul(W1, input_layer) + b1
	a2 = sigmoid(z2)

	output = np.matmul(W2, a2) + b2
	output = (sigmoid2(output))
	print (round(output))

