#!/usr/bin/python

import theano
import theano.tensor as T
import numpy as np
import time

from load import mnist

def floatX(X):  # convert to correct dtype
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):   # initialize model parameters
    return theano.shared(floatX(np.random.randn(*shape) * 0.001), borrow=True)

def model (X, w):   # model in matrix format
    return T.nnet.softmax(T.dot(X, w))

trX, teX, trY, teY = mnist(onehot=True)     # loading data matrices

X = T.fmatrix()
Y = T.fmatrix()

w = init_weights((784, 10)) # 784 = 28 * 28 size
py_x = model(X, w)
y_pred = T.argmax(py_x, axis=1) # probability outputs and maxima predictions

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))     # classification metric to optimize
gradient = T.grad(cost=cost, wrt=w)
updates = [[w, w - 0.05 * gradient]]

train = theano.function(inputs=[X, Y], outputs=[cost, y_pred], updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

start_time = time.time()
for i in range(100):
    start_epoch_time = time.time()
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        start_minibatch_time = time.time()
        cost, y_pred = train(trX[start:end], trY[start:end])
        end_minibatch_time = time.time()
    print("epoch : %d training time : %.6f sec. " % (i, (end_minibatch_time - start_time)))

    print i, np.mean(np.argmax(teY, axis=1) == predict(teX))



'''
for i in range(100):
    #cost_list = []
    for x, y in zip(trX, trY):
        cost, y_pred = train(x, y)
        cost_list.append(cost)
        mean_cost = np.mean(np.asarray(cost_list))
        print("epoch : %d, cost : %.6f, mean_cost : %f input(x) : %f, target(y) : %f, y_pred : %f" % (i, cost, mean_cost, x , y , y_pred)) 
    #print("epoch : %d, cost : %.6f, mean_cost : %f input(x) : %f, target(y) : %f, y_pred : %f" % (i, cost, mean_cost, x , y , y_pred)) 
'''    
