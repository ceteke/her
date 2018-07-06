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
        self.epsilon = 0
        self.experience_buffer = []
        self.sess = tf.Session()

        in_dim = 2*self.num_actions

        self.states = tf.placeholder(tf.float32, (None, in_dim), name='states')
        self.next_states = tf.placeholder(tf.float32, (None, in_dim), name='next_states')
        self.actions = tf.placeholder(tf.int32, self.minibatch_size, name='actions')
        self.rewards = tf.placeholder(tf.float32, self.minibatch_size, name='rewards')
        self.optimizer = tf.train.AdamOptimizer(self.lr)

        self.form_graph()

        self.sess.run(tf.global_variables_initializer())

    def q_net(self, inp, reuse=None):
        hidden = tf.layers.dense(inp, 256, tf.nn.relu, reuse=reuse, name='layer1')
        out = tf.layers.dense(hidden, self.num_actions, reuse=reuse, name='layer2')
        return out

    def form_graph(self):
        q_values = self.q_net(self.states)
        self.greedy = tf.argmax(q_values, axis=-1)

        actions_oh = tf.one_hot(self.actions, self.num_actions, dtype=tf.float32)
        selected_q = tf.reduce_sum(tf.multiply(actions_oh, q_values), axis=1)

        next_q = self.q_net(self.next_states, reuse=True)
        max_q = tf.reduce_max(next_q, axis=-1)
        target_q = self.rewards + self.discount * max_q
        target_q = tf.clip_by_value(target_q, -1 / (1 - self.discount), 0)

        self.error = tf.reduce_sum(tf.squared_difference(target_q, selected_q))
        self.update_op = self.optimizer.minimize(self.error)

    def store_transition(self, state, action, reward, next_state, goal):
        state = np.concatenate([state, goal], axis=0)
        next_state = np.concatenate([next_state, goal], axis=0)

        self.experience_buffer.append((state, action, reward, next_state))
        if len(self.experience_buffer) > self.buffer_size: del self.experience_buffer[0]

    def get_action(self, state, goal=None):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions-1)
        inp = np.concatenate([state, goal], axis=0).reshape(1,-1)
        return self.sess.run(self.greedy, feed_dict={self.states:inp})[0]

    def update(self):
        try:
            minibatch = random.sample(self.experience_buffer, self.minibatch_size)
        except ValueError:
            return

        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])

        self.sess.run(self.update_op, feed_dict={self.states:states,
                                                 self.actions:actions,
                                                 self.rewards:rewards,
                                                 self.next_states:next_states})