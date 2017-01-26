#!/usr/bin/env python

"""
A quick start for Tensorflow
Computes y = \sum{W * x}
Adapted for tensorflow from http://www.marekrei.com/blog/theano-tutorial/
"""

import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, shape=(2,), name='x')

W = tf.Variable(tf.constant([0.2, 0.7]), name='W')

y = tf.reduce_sum(x * W)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print sess.run(y, feed_dict={x: [1.0, 1.0]})
