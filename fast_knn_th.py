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
import theano
import theano.tensor as T
import time
import os
from sklearn.neighbors import KNeighborsClassifier
from keras.datasets import mnist

# Import MNIST data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = np.reshape(X_train, (60000, 784))
X_test = np.reshape(X_test, (10000, 784))

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.
X_test = X_test / 255.

X_test = X_test[:1000]
Y_test = Y_test[:1000]


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


def l2_normalize(x, dim, epsilon=1e-12):
    square_sum = T.sum(T.sqr(x), axis=dim)
    x_inv_norm = T.true_div(1, T.sqrt(T.maximum(square_sum, epsilon)))
    x_inv_norm = x_inv_norm.dimshuffle(0, 'x')
    return T.mul(x, x_inv_norm)

### Theano KNN ###
# Construct the theano graph
x_keys = T.matrix('x_keys')
x_queries = T.matrix('x_keys')

normalized_keys = l2_normalize(x_keys, dim=1)
normalized_query = l2_normalize(x_queries, dim=1)
query_result = T.dot(normalized_keys, normalized_query.T)
pred = T.argmax(query_result, axis=0)

# Declare the knn function
knn = theano.function(inputs=[x_keys, x_queries], outputs=pred)

print("Evaluating theano KNN")
start = time.time()
nn_index = knn(X_train, X_test)
end = time.time()

accuracy = np.sum(Y_train[nn_index] == Y_test).astype(float) / len(Y_test)

print("Accuracy: " + str(accuracy * 100) + '%')
print('Took', end - start, 'seconds')
