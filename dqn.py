import tensorflow as tf
import random
import numpy as np


class DQN(object):
    def __init__(self, num_actions, her=True):
        self.num_actions = num_actions
        self.discount = 0.98
        self.buffer_size = int(1e6)
        self.minibatch_size = 128
        self.lr = 1e-3
        self.experience_buffer = []
        self.sess = tf.Session()

        in_dim = 2*self.num_actions if her else self.num_actions

        self.states = tf.placeholder(tf.float32, (self.minibatch_size, in_dim))
        self.actions = tf.placeholder(tf.int32, self.minibatch_size)
        self.target_q = tf.placeholder(tf.float32, self.minibatch_size)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.form_graph()

        self.sess.run(tf.global_variables_initializer())

    def form_graph(self):
        self.hidden = tf.layers.dense(self.states, 256, activation=tf.nn.relu)
        self.q_values = tf.layers.dense(self.hidden, self.num_actions)
        self.greedy = tf.argmax(self.q_values, axis=-1)

        actions_oh = tf.one_hot(self.actions, self.num_actions, dtype=tf.float32)
        selected_q = tf.reduce_sum(tf.multiply(actions_oh, self.q_values), axis=1)

        self.error = tf.reduce_sum(tf.squared_difference(self.target_q, selected_q))
        self.update_op = self.optimizer.minimize(self.error)

    def store_transition(self):
        self.experience_buffer.append(None)
        if len(self.experience_buffer) > self.buffer_size: del self.experience_buffer[0]

    def update(self):
        pass
        # minibatch = random.sample(self.experience_buffer, self.minibatch_size)
