## Learning simple AND operation using logistic regression

import numpy as np
import math

import matplotlib.pyplot as plt

def sigmoid(X):
	return 1/(1 + math.exp(-X))

def round(X):
	if X >= 0.5:
		return 1

	else: 
		return 0
		
## Initializing X to inputs of AND
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

## Adding the bias element to X
X= np.insert(X, 0, np.ones(X.shape[0]), axis=1)
y = np.array([0, 0, 0, 1])

X = X.T

## Initializing the weights to 0 initially
W = np.zeros(len(X))

## Setting the bias of weight as 1
W[0] = 1

## Performing gradient descent
for epoch in range(1000):
	pred = np.matmul(W.T, X)
	for i in range(len(pred)):
		pred[i] = sigmoid(pred[i])

	for j in range(len(W)):
		gradient = 0
		for i in range(X.shape[1]):
			gradient += (pred[i] - y[i]) * X[j, i]

		W[j] -= 0.1 * gradient

## Making the predictions
pred = np.matmul(W.T, X)
for i in range(len(pred)):
	pred[i] = round(sigmoid(pred[i]))

print (pred, y, 'Weights = ', W)

## Plotting the decision boundary
X_plot = [-3, 3]
y_plot = [-1 * W[1]/W[2] * i -1 * W[0]/W[2] for i in X_plot]
plt.plot(X_plot, y_plot, '-')
plt.plot(X[1], X[2], 'ro')
plt.plot(X[1][3], y[3], 'go')
plt.show()

