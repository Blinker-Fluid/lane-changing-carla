import random
import numpy as np
import glob
import os
import cv2
import math
import matplotlib.pyplot as plt
from collections import deque
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, concatenate, Conv2D, AveragePooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard
from tqdm import tqdm
# import keras.backend.tensorflow_backend as backend
# from threading import Thread
# import tensorflow as tf
import tensorflow.compat.v1 as tf
# tf.compat.v1.disable_v2_behavior()
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
# from carla_env.new_env import *
tf.disable_v2_behavior()

class DQNAgent:

    def __init__(self, state_height, state_width, action_size):
        self.state_height = state_height
        self.state_width = state_width
        self.action_size = action_size
        self.memory1 = deque(maxlen=20000)
        self.memory2 = deque(maxlen=20000)
        # self.memory3 = deque(maxlen=20000)
        self.gamma = 0.90    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.3
        self.epsilon_decay = 0.9  # init with pure exploration
        self.learning_rate = 0.00025
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        # input1 = Input(shape=(1, self.state_height, self.state_width))
        # conv1 = Conv2D(64, (4, 2), strides=1, activation='relu', padding='valid', data_format='channels_first',
        #                input_shape=(1, self.state_height, self.state_width))(input1)
        # conv2 = Conv2D(64, (4, 2), strides=1, activation='relu',
        #                padding='valid')(conv1)
        # conv3 = Conv2D(3, 1, strides=1, activation='relu',
        #                padding='valid')(conv2)
        # state1 = Flatten()(conv3)
        # input2 = Input(shape=(3,))
        # state2 = concatenate([input2, state1])
        # state2 = Dense(256, activation='relu')(state2)
        # state2 = Dense(64, activation='relu')(state2)
        # out_put = Dense(self.action_size, activation='linear')(state2)
        # model = Model(inputs=[input1, input2], outputs=out_put)
        # model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        model = Sequential()
        model.add(Conv2D(64, (4, 2), strides=1, activation='relu', padding='valid', data_format='channels_first',
                         input_shape=(1, self.state_height, self.state_width)))
        model.add(Conv2D(64, (4, 2), strides=1, activation='relu', padding='valid'))
        model.add(Conv2D(3, 1, strides=1, activation='relu', padding='valid'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember1(self, state, action, reward, next_state, done):
        self.memory1.append((state, action, reward, next_state, done))

    def remember2(self, state, action, reward, next_state, done):
        self.memory2.append((state, action, reward, next_state, done))

    def remember3(self, state, action, reward, next_state, done):
        self.memory3.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            print('random')
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch1 = random.sample(self.memory1, int(batch_size / 2))
        minibatch2 = random.sample(self.memory2, batch_size - int(batch_size / 2))
        minibatch = minibatch1 + minibatch2

        for state, action, reward, next_state, done in minibatch:
            # if not done:
            # print(state.shape)
            next_state = np.reshape(next_state, [-1, 1, self.state_height, self.state_width])
            # print(next_state.shape)
            # pos = [0, 0, 0]
            # pos = np.reshape(pos, [1, 3])
            # target = self.model.predict((state, pos))
            target = self.model.predict(state)
            t = self.target_model.predict(next_state)[0]
            target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
            # else:
            #     target = self.model.predict(state)
            #     target[0][action] = reward

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)