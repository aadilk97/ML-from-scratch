import tflearn
import numpy as np
import sklearn
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder
from tflearn.datasets import cifar10
from tflearn.data_utils import to_categorical

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

X_train = np.float32(np.array(X_train[1:10000]))
Y_train = np.float32(np.array(Y_train[1:10000]))

Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)



X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

## Convolution 1
W1 = tf.Variable(tf.random_normal([3, 3, 3, 32]), dtype=tf.float32, name='W1')
b1 = tf.Variable(tf.random_normal([32]), dtype=tf.float32, name='b1')

x1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='VALID')
x1 = tf.add(x1, b1)
x1 = tf.nn.relu(x1)


## Max pooling 1
x1 = tf.nn.max_pool(x1, [1,2,2,1], strides=[1,1,1,1], padding='VALID')


## Convolution 2
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64]), dtype=tf.float32, name='W2')
b2 = tf.Variable(tf.random_normal([64]), dtype=tf.float32, name='b2')

x2 = tf.nn.conv2d(x1, W2, strides=[1,1,1,1], padding='VALID')
x2 = tf.add(x2, b2)
x2 = tf.nn.relu(x2)


## Convolution 3
W3 = tf.Variable(tf.random_normal([3, 3, 64, 64]), dtype=tf.float32, name='W3')
b3 = tf.Variable(tf.random_normal([64]), dtype=tf.float32, name='b3')

x3 = tf.nn.conv2d(x2, W3, strides=[1,1,1,1], padding='VALID')
x3 = tf.add(x3, b3)
x3 = tf.nn.relu(x3)


## Max pooling 2
x3 = tf.nn.max_pool(x3, [1,2,2,1], strides=[1,1,1,1], padding='VALID')


## Fully connected 1
W4 = tf.Variable(tf.random_normal([512, 36864]), dtype=tf.float32, name='W4')
b4 = tf.Variable(tf.random_normal([512]), dtype=tf.float32, name='b4')

x3 = tf.reshape(x3, [-1, 36864])
fully_connected1 = tf.matmul(x3, tf.transpose(W4))
fully_connected1 = tf.add(fully_connected1, b4)
fully_connected1 = tf.nn.relu(fully_connected1)

##Drop out
fully_connected1 = tf.nn.dropout(fully_connected1, 0.5)

## Output 
W5 = tf.Variable(tf.random_normal([10, 512]), dtype=tf.float32, name='W5')
b5 = tf.Variable(tf.random_normal([10]), dtype=tf.float32, name='b5')

y_ = tf.matmul(fully_connected1, tf.transpose(W5))
y_ = tf.add(y_, b5)

output = tf.nn.softmax(y_)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)


init_g = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init_g)
	# xt, _ = sess.run([cost, optimizer], feed_dict={X: X_train, y:Y_train})

	batch_size = 100
	for epoch in range(2):
		curr_pointer = 0
		epoch_cost = 0
		while curr_pointer < (X_train.shape[0]):
			batch_x = np.float32(np.array(X_train[curr_pointer: curr_pointer + batch_size]))
			batch_y = np.float32(np.array(Y_train[curr_pointer: curr_pointer + batch_size]))
			curr_pointer += batch_size

			_, c = sess.run([optimizer, cost], feed_dict={X: batch_x, y:batch_y})
			epoch_cost += c

			print (curr_pointer)

		print ("Cost for epoch ", epoch, "= ", epoch_cost)

	

# x2 = tflearn.layers.conv.conv_2d(X_train, 10, 3, padding='valid')
# print (x2.shape)
