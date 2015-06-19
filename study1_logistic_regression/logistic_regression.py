#!/usr/bin/python

import theano
import theano.tensor as T
import numpy as np

trX = np.linspace(-1, 1, 101)
trY = 2 * trX - np.random.randn(*trX.shape) * 0.33

print("trX : ", trX)
print("trY : ", trY)

X = T.scalar()
Y = T.scalar()

def model(X, w):
    #return X * w
    #return T.tanh(X * w)
    return T.minimum(0, X * w)

w = theano.shared(np.asarray(0, dtype=theano.config.floatX))
y = model(X, w)

cost = T.mean(T.sqr(y - Y))
gradient = T.grad(cost=cost, wrt=w)
updates = [[w, w - 0.01 * gradient]]

train = theano.function(inputs=[X, Y], outputs=[cost, y], updates=updates, allow_input_downcast=True)

for i in range(100):
    cost_list = []
    for x, y in zip(trX, trY):
        cost, y_pred = train(x, y)
        cost_list.append(cost)
        mean_cost = np.mean(np.asarray(cost_list))
        print("epoch : %d, cost : %.6f, mean_cost : %f input(x) : %f, target(y) : %f, y_pred : %f" % (i, cost, mean_cost, x , y , y_pred)) 
    #print("epoch : %d, cost : %.6f, mean_cost : %f input(x) : %f, target(y) : %f, y_pred : %f" % (i, cost, mean_cost, x , y , y_pred)) 
    
