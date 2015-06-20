#!/bin/bash

#mkdir -p /home/kwangho/research/Project/DeepLearning_Theano/mnist
mkdir -p /home/kwangho/research/Project/DeepLearning_Theano/mnist
if ! [ -e /home/kwangho/research/Project/DeepLearning_Theano/mnist/train-images-idx3-ubyte.gz ]
	then
		wget -P /home/kwangho/research/Project/DeepLearning_Theano/mnist/ http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
fi
gzip -d /home/kwangho/research/Project/DeepLearning_Theano/mnist/train-images-idx3-ubyte.gz

if ! [ -e /home/kwangho/research/Project/DeepLearning_Theano/mnist/train-labels-idx1-ubyte.gz ]
	then
		wget -P /home/kwangho/research/Project/DeepLearning_Theano/mnist/ http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
fi
gzip -d /home/kwangho/research/Project/DeepLearning_Theano/mnist/train-labels-idx1-ubyte.gz

if ! [ -e /home/kwangho/research/Project/DeepLearning_Theano/mnist/t10k-images-idx3-ubyte.gz ]
	then
		wget -P /home/kwangho/research/Project/DeepLearning_Theano/mnist/ http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
fi
gzip -d /home/kwangho/research/Project/DeepLearning_Theano/mnist/t10k-images-idx3-ubyte.gz

if ! [ -e /home/kwangho/research/Project/DeepLearning_Theano/mnist/t10k-labels-idx1-ubyte.gz ]
	then
		wget -P /home/kwangho/research/Project/DeepLearning_Theano/mnist/ http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
fi
gzip -d /home/kwangho/research/Project/DeepLearning_Theano/mnist/t10k-labels-idx1-ubyte.gz
