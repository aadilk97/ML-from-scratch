import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression


def sigmoid(gamma):
  if gamma < 0:
    return 1 - 1/(1 + math.exp(gamma))
  else:
    return 1/(1 + math.exp(-gamma))

def round(X):
	if X >= 0.5:
		return 1

	else: 
		return 0

def score(pred, target):
	score = 0
	for i in range(len(pred)):
		if pred[i] == target[i]:
			score += 1

	return score / len(pred)

## Read the data into pandas dataframe
df = pd.read_csv('data.csv', delimiter=',')

## Get the taregt labels and the features
y = np.array(df[df.columns[1]])
X = np.array(df.drop(df.columns[[0, 1]], axis=1))

## Normalizing the input X
for i in range (X.shape[0]):
	X[i] /= np.max(X[i])

## Encode the target labels
le = LabelEncoder()
le = le.fit(y)
y = le.transform(y)
y = y.reshape((len(y), 1))


## Split the data into 80% training set and 20% test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## Adding the bias to the input sets
X_train = np.insert(X_train, 0, np.ones(X_train.shape[0]), axis=1)
X_test = np.insert(X_test, 0, np.ones(X_test.shape[0]), axis=1)

X_train = X_train.T
X_test = X_test.T

## Initiliazing the weights
W = np.zeros(len(X_train))
W[0] = 1

num_epochs = 500
learning_rate = 0.01

## Performing basic SGD
for epoch in range(num_epochs):
	pred = np.matmul(W.T, X_train)
	for i in range(len(pred)):
		pred[i] = sigmoid(pred[i])

	epoch_error = 0
	for j in range(len(W)):
		for i in range(X_train.shape[1]):
			## Calculating the gradient for each training instance 
			graident = (pred[i] - y_train[i]) * X_train[j, i]

			## Updating the weights
			W[j] -= learning_rate * graident

	print ("Epoch = ", epoch)

## Making the prediction
pred = np.matmul(W.T, X_test)
for i in range(len(pred)):
	pred[i] = round(sigmoid(pred[i]))

print ("Accuracy = ", score(pred, y_test))

model = LogisticRegression()
model.fit(X_train.T, y_train)
print ("sklearn accuracy = ", model.score(X_test.T, y_test))


