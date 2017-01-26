#!/usr/bin/env python

"""
Line fitting with tensorflow

With randomly generated data points for the line y = x * 0.1 + 0.3 ,
we will try to learn W, b for y = W * x + b
to minimize the mean squared error over the dataset.

This tutorial was adapted from : https://www.tensorflow.org/get_started/
with additional instructions for the JHU Neural Winter School

"""

import tensorflow as tf
import numpy as np

# Create 100 data points x,y in Numpy that fit y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Create tensorflow variables and initialize them with some sensible values
# Remember, we are trying to find the optimal W, b to fit a line to the data
# we just generated.
# We'll use tensorflow to help us do that.
# Note the use of the numpy like helper functions we use to initiate these variables.
# W is a scalar value (a 1 dimensional 1-tensor)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# and so is b
b = tf.Variable(tf.zeros([1]))

# This is a symbolic expression for a mathematical computation (a couple of them, actually)
# These are expanded in the computation graph
y = W * x_data + b

# This is the loss function
# We will use mean squared error and minimize it over the dataset
# y is our prediction, y_data contains the truth
# L = \sum_D (y - y_{data})^2
loss = tf.reduce_mean(tf.square(y - y_data))

# We will use gradient descent with a learning rate of 0.5 to learn the weights
# Tensorflow has us covered
# -> Automatically compute gradients wrt parameters
# -> Minimize loss
# -> Update parameters

# First, initialize the optimizer
optimizer = tf.train.GradientDescentOptimizer(0.5)
# Now define the operation for minimizing the loss
train = optimizer.minimize(loss)

# Time to get to work
# First initialize all variables (put values in variable containers we defined above)
# Always run this before building the computation graph
init = tf.global_variables_initializer()

# Create and launch the graph now
sess = tf.Session()
sess.run(init)

# Here's the actual training procedure
# We will run 200 steps (epochs in this case)
for step in range(201):
  # This will evaluate the final node in the computation graph
  # Note that because of the dependencies, this will work backwards and
  # evaluate all other operations
  sess.run(train)
  # Check to see how the training is doing after every 20 steps
  if step % 20 == 0:
    # Note how we used run to fetch the value of the variable?
    # We did the same thing to train
    # Think of this as a top down traversal of the computation graph
    # When we want to evaluate something, Tensorflow will evaluate all dependent
    # expressions in the computaion graph
    print(step, sess.run(W), sess.run(b))

# How did we do?
#(0, array([-0.30659565], dtype=float32), array([ 0.69793773], dtype=float32))
#(20, array([ 0.00509692], dtype=float32), array([ 0.34970179], dtype=float32))
#(40, array([ 0.08133842], dtype=float32), array([ 0.3097733], dtype=float32))
#(60, array([ 0.09633042], dtype=float32), array([ 0.30192181], dtype=float32))
#(80, array([ 0.09927841], dtype=float32), array([ 0.30037791], dtype=float32))
#(100, array([ 0.09985811], dtype=float32), array([ 0.30007431], dtype=float32))
#(120, array([ 0.0999721], dtype=float32), array([ 0.30001462], dtype=float32))
#(140, array([ 0.09999453], dtype=float32), array([ 0.30000287], dtype=float32))
#(160, array([ 0.09999894], dtype=float32), array([ 0.30000058], dtype=float32))
#(180, array([ 0.09999979], dtype=float32), array([ 0.30000013], dtype=float32))
#(200, array([ 0.09999991], dtype=float32), array([ 0.30000007], dtype=float32))
