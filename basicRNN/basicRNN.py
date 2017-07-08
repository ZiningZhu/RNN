import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import os
import re
import pandas as pd

from vocab2vec import *


# Adapted from https://gist.github.com/muggin/3097e7ed45a75dd53bd96c0e430a2895
class TFBasicRNN(object):
    def __init__(self, vocab_size, hidden_size, seq_length):
        # size of vocabulary
        self.vocab_size = vocab_size
        # size of hidden state
        self.hidden_size = hidden_size
        # length of input sequence
        self.seq_length = seq_length
        # tf computation graph
        self.graph = tf.Graph()

        # define and setup tf graph nodes
        self._create_constants()
        self._create_variables()
        self._create_placeholders()
        self._create_loss_and_optimizer()

        self.ix_to_char = 'abcdefghijklmnopqrstuvwxyz .,\'1234567890";'


    def _create_placeholders(self):
        with self.graph.as_default():
            with self.graph.name_scope('placeholders'):
                self.inputs = tf.placeholder(shape=[self.seq_length, self.vocab_size], dtype=tf.float32, name='inputs')
                self.inputs_shape = tf.shape(self.inputs)
                self.targets = tf.placeholder(shape=[self.seq_length, self.vocab_size], dtype=tf.float32, name='targets')
                self.init_state = tf.placeholder(shape=[1, self.hidden_size], dtype=tf.float32, name='state')

    def _create_constants(self):
        with self.graph.as_default():
            with tf.name_scope('constants'):
                self.grad_limit = tf.constant(5.0, dtype=tf.float32, name='grad_limit')

    def _create_variables(self):
        with self.graph.as_default():
            with tf.name_scope('weights'):
                self.Wxh = tf.Variable(tf.random_normal(stddev=0.1, shape=(self.vocab_size, self.hidden_size)), name='Wxh')
                self.Whh = tf.Variable(tf.random_normal(stddev=0.1, shape=(self.hidden_size, self.hidden_size)), name='Whh')
                self.Why = tf.Variable(tf.random_normal(stddev=0.1, shape=(self.hidden_size, self.vocab_size)), name='Why')
                self.bh = tf.Variable(tf.zeros((self.hidden_size)), name='bh')
                self.by = tf.Variable(tf.zeros((self.vocab_size)), name='by')

    def _create_loss_and_optimizer(self):
        self.grad_limit = 100 # tentative

        with self.graph.as_default():
            with tf.name_scope('loss'):
                hs_t = self.init_state
                ys = []
                # forward pass through the network.
                for t, xs_t in enumerate(tf.split(self.inputs, self.seq_length, axis=0)):
                    hs_t = tf.tanh(tf.matmul(xs_t, self.Wxh) + tf.matmul(hs_t, self.Whh) + self.bh)

                    ys_t = tf.matmul(hs_t, self.Why) + self.by
                    ys.append(ys_t)
                outputs = tf.concat(ys, axis=0)

                # update state after forward pass
                self.update_state = hs_t

                # transform network output to probabilities
                self.outputs_softmax = tf.nn.softmax(ys[-1])

                # define loss
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.targets, logits=outputs))

                # compute and update gradients
                self.optimizer = tf.train.AdamOptimizer()
                grads_and_vars = self.optimizer.compute_gradients(self.loss)
                clipped_grads_and_vars = []
                for grad, var in grads_and_vars:
                    clipped_grad = tf.clip_by_value(grad, -self.grad_limit, self.grad_limit)
                    clipped_grads_and_vars.append((clipped_grad, var))
                self.updates = self.optimizer.apply_gradients(clipped_grads_and_vars)



    def train(self, text, max_iter=100001, sample_dist=True):
        # pointer to input data
        p = 0

        plt_iter = []
        plt_loss = []
        with tf.Session(graph=self.graph) as sess:
            # init variables
            sess.run(tf.global_variables_initializer())

            n = 0
            while (n < max_iter):
                # do data bookkeeping
                if p + self.seq_length + 1 >= len(text) or n == 0:
                    # reset hidden state and data pointer
                    p = 0
                    state_prev = np.zeros((1, self.hidden_size))
                    n += 1


                inputs_ohe = text[p:p+self.seq_length, :]
                targets_ohe = text[p+1:p+self.seq_length+1, :]

                # training step
                feed_data = {self.inputs: inputs_ohe, self.targets: targets_ohe, self.init_state: state_prev}
                current_state, loss, _ = sess.run([self.update_state, self.loss, self.updates], feed_dict=feed_data)
                state_prev = np.copy(current_state)

                plt_iter.append(n)
                plt_loss.append(loss)
                if n % 10 == 0:
                    print ('step: %d - p: %d -- loss: %f' % (n, p, loss))



                if sample_dist and n % 10 == 0:
                    # sample a random sequence
                    sample_length = 200
                    start_ix = np.random.randint(0, len(text) - self.seq_length)

                    sample_input_vals = text[start_ix : start_ix + self.seq_length, :]
                    ixes = []
                    sample_state_prev = np.copy(current_state)



                    for t in range(sample_length):
                        #print ("shape of sample_input_vals: " + str(sample_input_vals.shape))
                        feed_data = {self.inputs: sample_input_vals, self.init_state: sample_state_prev}
                        #print ("Testing: self.inputs shape is {}".format(
                        #    sess.run(self.inputs_shape)
                        #))
                        sample_output_softmax_val, sample_current_state = \
                            sess.run([self.outputs_softmax, self.update_state], feed_dict=feed_data)
                        ix = np.random.choice(range(self.vocab_size), p=sample_output_softmax_val.ravel())
                        # Since the output is probability, ix is actually the choice of index!

                        ixes.append(ix)


                        sample_input_vals = np.append(sample_input_vals[1:], sample_output_softmax_val, axis=0)

                    txt = ''.join(self.ix_to_char[ix] for ix in ixes)
                    print('----\n %s \n----\n' % (txt,))
                p += self.seq_length


        plt.plot(plt_iter, plt_loss)
        plt.title("Char-RNN training loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.savefig("training_curve.png")
