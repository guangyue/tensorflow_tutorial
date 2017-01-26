#!/usr/bin/env python

"""
A quick start for Tensorflow
Computes y = \sum{W * x}
Adapted for tensorflow from http://www.marekrei.com/blog/theano-tutorial/
"""

import tensorflow as tf
import numpy as np

# A placeholder is tensorflow-talk for a container that will be filled later
# In this case, this is a container for values to be provided when evaluating y
# You need to specify the data type and shape for the values that will
# eventually end up in this placeholder
x = tf.placeholder(tf.float32, shape=(2,), name='x')

W = tf.Variable(tf.constant([0.2, 0.7]), name='W')

# A Variable is another type of container for values that is initialized with values
# You will typically use these for your model parameter
b = tf.Variable(tf.constant([0.2, 0.7]), name='W')

# A symbolic mathematical operation
y = tf.reduce_sum(x * W)

# First initialize all variables (put values in variable containers we defined above)
# Always run this before building the computation graph
init = tf.global_variables_initializer()

# Create and launch the graph now
sess = tf.Session()
sess.run(init)

# We want the value of y. We will use sess.run to get it.
# However, remember that y depends on x which is currently empty
# We will fill all dependecies (placeholders) for the expression
# we want to evaluate with a dictionary called feed_dict
# and pass it to sess.run()
print sess.run(y, feed_dict={x: [1.0, 1.0]})

# How did we do?
# python intro.py
# 0.9
