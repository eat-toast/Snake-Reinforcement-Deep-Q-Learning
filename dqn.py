import tensorflow as tf
import random
import numpy as np


# self.observe = 1000  # DQN 논문은 10만을 설정 했었음.

class DQN:
    def __init__(self, session, Action_size = 4, name = 'main'):
        # Initialize Variables
        # epoch = frame을 의미
        # episode = 게임 한판
        self.session = session
        self.action_size = Action_size
        self.epoch = 0
        self.episode = 0
        # discount factor
        self.discft = 0.999
        self.epsilon = 1.0

        self.finep = 0
        self.net_name = name

        self._bulid_ConvNet()

    def _bulid_ConvNet(self):
        with tf.variable_scope(self.net_name):

            # Init weight and bias
            self.w1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev=0.01))
            self.b1 = tf.Variable(tf.constant(0.01, shape=[32]))

            self.w2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
            self.b2 = tf.Variable(tf.constant(0.01, shape=[64]))

            self.wfc = tf.Variable(tf.truncated_normal([2304, 256], stddev=0.01))
            self.bfc = tf.Variable(tf.constant(0.01, shape=[256]))

            self.wto = tf.Variable(tf.truncated_normal([256, self.action_size], stddev=0.01))
            self.bto = tf.Variable(tf.constant(0.01, shape=[self.action_size]))

            # input layer
            self.input = tf.placeholder("float", [None, 84, 84, 4])

            # Convolutional Neural Network
            # zero-padding
            # 84 x 84 x 4
            # 8 x 8 x 4 with 32 Filters
            # Stride 4 -> Output 21 x 21 x 32 -> max_pool 11 x 11 x 32
            # tf.nn.conv2d(self.input, self.w1, strides = [1, 4, 4, 1], padding = "SAME")
            conv1 = tf.nn.relu(tf.nn.conv2d(self.input, self.w1, strides=[1, 4, 4, 1], padding="SAME") + self.b1)
            pool = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

            # 11 x 11 x 32
            # 4 x 4 x 32 with 64 Filters
            # Stride 2 -> Output 6 x 6 x 64
            conv2 = tf.nn.relu(tf.nn.conv2d(pool, self.w2, strides=[1, 2, 2, 1], padding="SAME") + self.b2)

            # 6 x 6 x 64 = 2304
            conv2_to_reshaped = tf.reshape(conv2, [-1, 2304])

            # Matrix (-1, 2304) * (2304, 256)
            fullyconnected = tf.nn.relu(tf.matmul(conv2_to_reshaped, self.wfc) + self.bfc)

            # output(Q) layer
            # Matrix (-1, 256) * (256, ACTIONS) -> (-1, ACTIONS)
            self._Qpred = tf.matmul(fullyconnected, self.wto) + self.bto  # Q_pred

            self._Y = tf.placeholder(dtype="float", shape = [None, self.action_size]) # optimal Q

            # loss function
            self.cost = tf.reduce_mean(tf.square(self._Y - self._Qpred))

            # Learning
            self._train = tf.train.AdamOptimizer(1e-6).minimize(self.cost)



    def predict(self, state):
        Q_batch = self._Qpred.eval(feed_dict={self.input: state})  # NN 에서 Q값을 계산한다.

        return Q_batch

    def update(self, x_stack, y_stack):
        self.session.run


    def getAction(self):
        Q_val = self._Qpred.eval(feed_dict={self.input: [self.s_t]})[0]
        # for print
        self.qv = Q_val

        # action array
        action = np.zeros(self.action_size)

        idx = np.argmax(Q_val)
        action[idx] = 1

        return action

    def initState(self, state):
        self.s_t = np.stack((state, state, state, state), axis=2)