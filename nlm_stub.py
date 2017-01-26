#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import sys


def create_ngram_data(input_file, ngram_size):
    '''Reads input_file and returns a character ngram dataset
    where each row is an ngram [char1, char2, char3,.. charN],
    with N=ngram_size, and the characters are mapped to ASCII id's.
    '''
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            for i in xrange(len(line) - ngram_size):
                data.append([ord(c) for c in line[i:i+ngram_size]])
    return data


def perplexity(sess, input_id, soft_output, traindata, ngram_size):
    '''Computes perplexity of the provided inference model on an ngram dataset
    '''
    nn = ngram_size - 1
    expn = 0.0
    count = 0
    for d in testdata:
        prob = sess.run(soft_output, feed_dict={input_id:d[0:nn]})[0][d[nn]]
        expn -= np.log2(prob)
        count +=1
    return 2**(expn/count)


def generate_sample(sess, input_id, soft_output, initial_input):
    '''Generate a random text sample given initial input and an inference model
    '''
    inp = initial_input
    print ' '.join([chr(i) for i in initial_input]),
    for i in range(30):
        #o = np.random.choice(128, 1, p=inference_model(inp)[0])[0]
        o = np.random.choice(128, 1, p=sess.run(soft_output, feed_dict={input_id:inp})[0])[0]
        #o = inference_model(inp)[0].argmax()
        print chr(o),
        inp.pop(0)
        inp.append(o)


ngram_size = 5
vocab_size = 128
embedding_size = 7
hidden_size = 25

# 0: run program: "python nlm.py textfile.txt" where textfile is an ASCII text    
input_text = sys.argv[1]

# 1: create dataset. we'll have 5-gram language models, i.e. given 4 characters, predict the 5th
data = create_ngram_data(input_text,ngram_size)
traindata, testdata = data[0:int(len(data)*0.8)], data[int(len(data)*0.8):]

'''Returns the (inference graph, training graph) of a feedforward neural language model
Assume a 5-gram setup, where input context = [char1,char2,char3,char4] to predict [char5]
- inference_graph([char1,char2,char3,char4]) outputs softmax probability distribution for char5
- train_graph([char1,char2,char3,char4],[char5]) will update neural language model and return training cost for this sample
'''

# embedding matrix 
E = tf.Variable(np.random.randn(vocab_size, embedding_size))

# input_id is a vector [char1,char2,char3,char4], represented by ASCII values
input_id = tf.placeholder(tf.int32, [ngram_size - 1])

W = tf.Variable(np.random.randn(embedding_size * (ngram_size - 1), hidden_size), 'W')
W_2 = tf.Variable(np.random.randn(hidden_size, vocab_size), 'W_2')
b = tf.Variable(np.random.randn(1, hidden_size), 'b')
b_2 = tf.Variable(np.random.randn(1, vocab_size), 'b_2')

params = [W, b, W_2, b_2]

# indexes the embedding matrix and concatenates
concat_embedding = tf.reshape(tf.gather(E, input_id), [1, -1])

# TODO1: MLP code here to connect concat_embedding to output
# HINT : Use tf.matmul for dot products
# HINT : Use tf.tanh for the tanh activation
# HINT : Use tf.nn.softmax to compute the softmax
....

# TODO2: code here for training. output_id is target to predict, i.e. char5
output_id = tf.placeholder(tf.int32, shape=())
# The symbolic expression for the loss that we will minimize
# HINT : Use tf.log to compute elementwise log
cost = ...
# HINT : Use tf.train.GradientDescentOptimizer(learning_rate) to initialize an opimizer
# HINT : Use optimizer.minimize(cost) to minimize the cost expression
train_step = ...

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 3: training loop
for epoch in range(10):
    cumulative_cost = 0
    for d in traindata:
        nn = ngram_size - 1
        c, _ =  sess.run([cost, train_step], feed_dict={input_id:d[0:nn], output_id:d[nn]})
        cumulative_cost += c
    print "Epoch=%d CumulativeCost=%f" %(epoch, cumulative_cost),
    print "TrainPerplexity=%f TestPerplexity=%f" % (perplexity(sess, input_id, soft_output, traindata, ngram_size),
                                                    perplexity(sess, input_id, soft_output, testdata, ngram_size))

for i in range(3):
    print "sample: ",
    generate_sample(sess, input_id, soft_output, data[i][0:nn])
    print ""
