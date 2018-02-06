import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

df = pd.read_csv('k_means_data.csv', delimiter='\t')
dataset = np.array(df)
labels = np.zeros(dataset.shape[0])

## Initializing K and randomly selecting k dataset points as initial clusters
k = 2
centroids = np.array([dataset[0], dataset[15]])

## Plotting the initial data points and centroids
plt.plot(dataset[:, 0], dataset[:, 1], 'ro')
plt.plot(centroids[:, 0], centroids[:, 1], 'yo')
plt.show()

for epoch in range(10):
	## Finding the closest centroid for each data point and assigning it to that cluster
	for i in range(dataset.shape[0]):
		distance = np.zeros(k)
		for j in range(k):
			distance[j] = math.sqrt(np.sum(np.power((np.subtract(dataset[i], centroids[j])), 2)))

		labels[i] = np.argmin(distance)

	## Updating the centroids 
	for j in range(k):
		centroids[j]= np.mean(np.array([dataset[i] for i in range(dataset.shape[0]) if labels[i] == j]), axis = 0)

	print (centroids)

	## Plotting the arrangement after each iteration
	colors = ['ro', 'go', 'bo', 'co', 'mo', 'yo', 'ko', 'wo']
	for i in range(dataset.shape[0]):
		for j in range(k):
			if labels[i] == j:
				plt.plot(dataset[i, 0], dataset[i, 1], colors[j])

	plt.plot(centroids[:, 0], centroids[:, 1], 'yo')
	plt.show()

## Printing out the final labels
print (labels)