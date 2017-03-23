'''
Super fast k-nearest-neighbour search for GPU
Based on the method used in "Learning To Remember Rare Events"
by Lukasz Kaiser, Ofir Nachun, Aurko Roy, and Samy Bengio
Paper: https://openreview.net/pdf?id=SJTQLdqlg
Code: https://github.com/tensorflow/models/tree/master/learning_to_remember_rare_events

Takes ~560 seconds for sklearn implementation to find 10,000 nearest neighbours
Takes ~0.17 seconds for Tensorflow to find 10,000 nearest neighbours (using GTX Titan X Pascal)
>3000x speed up!
'''

from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import os
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.examples.tutorials.mnist import input_data

# Restrict tensorflow to only use the first GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import MNIST data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

# Load in all samples
X_train, Y_train = mnist.train.next_batch(55000)
X_test, Y_test = mnist.test.next_batch(100)


### sklearn KNN ###
print("Evaluating sklearn KNN")
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train, Y_train)
start = time.time()
pred = neigh.predict(X_test)
end = time.time()

accuracy = np.sum(pred == Y_test).astype(float) / len(Y_test)
print("Accuracy: " + str(accuracy * 100) + '%')
print('Took', end - start, 'seconds')


### tensorflow KNN ###
# Construct the tensorflow graph
x_keys = tf.placeholder("float", [None, 784])
x_queries = tf.placeholder("float", [None, 784])

normalized_keys = tf.nn.l2_normalize(x_keys, dim=1)
normalized_query = tf.nn.l2_normalize(x_queries, dim=1)
query_result = tf.matmul(normalized_keys, tf.transpose(normalized_query))
pred = tf.arg_max(query_result, dimension=0)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    start = time.time()
    print("Evaluating tensorflow KNN")
    nn_index = sess.run(pred, feed_dict={x_keys: X_train, x_queries: X_test})
    end = time.time()

    accuracy = np.sum(Y_train[nn_index] == Y_test).astype(float) / len(Y_test)

    print("Accuracy: " + str(accuracy * 100) + '%')
    print('Took', end - start, 'seconds')
