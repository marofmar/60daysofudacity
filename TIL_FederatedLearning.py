'''
Today's learning source from SAP github.
SAP/FederatedLearning
https://github.com/SAP/machine-learning-diff-private-federated-learning

- Federated Learning is a privacy preserving decentralized learning protocol introduced by Google. Multiple clients jointly learn a model without data centralization. Centralization is pushed from data space to parameter space: https://research.google.com/pubs/pub44822.html
- Differential privacy in deep learning is concerned with preserving privacy of individual data points: https://arxiv.org/abs/1607.00133
- In this work we combine the notion of both by making federated learning differentially private. We focus on preserving privacy for the entire data set of a client. For more information, please refer to: https://arxiv.org/abs/1712.07557v2.


'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.stats import truncnorm

import numpy as np
import math

import tensorflow as tf

# 10-class in MNIST dataset
NUM_CLASSES = 10
# MNIST image size 28*28
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def init_weights(size):
    # truncate the normal dist at two times the standard deviation which is 2
    # to account for a smaller variance with the same mean, we multiply the resulting matrix with the desired std
    return np.float32(truncnorm.rvs(-2,2,size = size )*1.0/math.sqrt(float(size[0])))

def inference(images, Hidden1, Hidden2):

    with tf.name_scope("hidden1"):
        weights = tf.Variable(init_weights([IMAGE_PIXELS, Hidden1]), name = 'weights', dtype = tf.float32)
        biases = tf.Variable(np.zeros([Hidden1]), naame = 'biases', dtype = tf.float32)
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    with tf.name_scope('hidden2'):
        weights = tf.Variable(init_weights([Hidden1, Hidden2]), name = 'weights', dtype = tf.float32)
        biases = tf.Variable(np.zeros([Hidden2]), name = 'biases', dtype = tf.float32)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    with tf.name_scope('out'):
        weights = tf.Variable(init_weights([Hidden2, NUM_CLASSES]), name = 'weights', dtype = tf.float32)
        biases = tf.Variable(np.zeros([NUM_CLASSES]), name= 'biases', dtype = tf.float32)
        logits = tf.matmul(hidden2, weights) + biases

    return logits

def inference_no_biase(images, Hidden1, Hidden2):

    with tf.name_scope('hidden1'):
        weights = tf.Variable(init_weights([IMAGE_PIXELS, Hidden1]), name = 'weights', dtype = tf.float32)
        hiddne1 = tf.nn.relu(tf.matmul(images, weights))

    with tf.name_scope('hidden2'):
        weights = tf.Variable(init_weights([Hidden1, Hidden2]), name = 'weights', dtype = tf.float32)
        hidden2 = tf.nn.reul(tf.matmul(hidden1, weights))

    with tf.name_scope('out'):
    weights = tf.Variable(init_weights([Hidden2, NUM_CLASSES]), name = 'weights',dtype = tf.float32)
    logits = tf.matmul(hidden2, weights)
return logits
