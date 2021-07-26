#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 15:06:46 2021

@author: timhartnett
"""
import pandas as pd #stable version 1.3.0
import os
import gpflow #stable version 2.2.1
import numpy as np #stable version 1.19.2
from tensorflow import keras #stable version 2.5.0
import tensorflow as tf #stable version 2.5.0
from gpflow.utilities import print_summary
import random
from collections import deque

random.seed(42)
WORKING_DIR = os.getcwd()
data_df = pd.read_csv(WORKING_DIR+'/train.csv')
data = data_df.values
initial_set = data[random.sample(range(data.shape[0]),100),]
X = initial_set[:,:80].reshape(-1,80)
Y = initial_set[:,81].reshape(-1,1)

k = gpflow.kernels.Matern52()

m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)

opt = gpflow.optimizers.Scipy()

opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
print_summary(m)

class GP_model(object):
    '''
    model for bayesian prediction of material properties
    '''
    def __init__(self,training_data):
        self.training_data = training_data
    def train_GP(self,kernel='Matern52',optimizer='L-BFGS-B'):
        X = self.training_data[:,:-1]
        Y = self.training_data[:,-1].reshape(-1,1)
        k = gpflow.kernels.Matern52()
        m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)
        opt = gpflow.optimizers.Scipy()
        self.opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
        return m

class environment(object):
    '''
    Environment is described by normalized svalues of training value (y-mu)/sd. top 20 and bottom 20
    '''
    def __init__(self,virtual_data):
        self.virtual_data = virtual_data
    def state(self,current_data):
        mean = np.mean(current_data[:,-1])
        sd = np.std(current_data[:,-1])
        norm_data = (current_data[:,-1]-mean)/sd
        norm_data.sort()
        state = np.concatenate((norm_data[:20],norm_data[-20:]))
        return state
    def reset(self, n_initial):
        '''
        this method is to generate a random initial training set for a toy problem and train the initial GP
        '''
        self.current_data = self.virtual_data[random.sample(range(data.shape[0]),n_initial),:]
        self.current_state = self.state(self.current_data)
        model = GP_model(self.current_data)
        self.m = model.train_GP()
    def predict_virtual(self):
        mean, sd = self.m.predict_y(self.virtual_data[:,:-1])
        return mean, sd
    def step(self,action):
        mean, var = self.predict_virtual()
        if action == 0:
            info = 'exploit'
            new_data_index = np.argmax(mean)
            new_data = self.virtual_data[new_data_index,:].reshape((1,self.current_data.shape[1]))
            self.current_data = np.vstack((self.current_data,new_data))
        elif action == 1:
            info = 'explore'
            new_data_index = np.argmax(var)
            new_data = self.virtual_data[new_data_index,:].reshape((1,self.current_data.shape[1]))
            self.current_data = np.vstack((self.current_data,new_data))
        else:
            info = 'random'
            new_data = self.virtual_data[random.sample(range(self.virtual_data.shape[0]),k=1),:]
            self.current_data = np.vstack((self.current_data,new_data))
        new_state = self.state(self.current_data)
        reward = -1
        done = self.is_done()
        self.current_state = new_state
        return new_state, reward, done, info
    def is_done(self):
        if np.argmax(self.virtual_data[:,-1]) in self.current_data:
            done = True 
        else:
            done = False
        return done
    def update_model(self):
        model = GP_model(self.current_data)
        self.m = model.train_GP()

class agent:
    """ 
    The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
    """
    def __init__(self,state_shape, action_shape,learning_rate):
        self.learning_rate = learning_rate
        self.state_shape = state_shape
        self.action_shape = action_shape
    def DQN(self):
        init = tf.keras.initializers.HeUniform()
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_shape=self.state_shape, activation='relu',
                                     kernel_initializer=init))
        model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(self.action_shape, activation='linear', kernel_initializer=init))
        model.compile(loss=tf.keras.losses.Huber(),
                           optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model
        
def get_qs(model, state, step):
    return model.predict(state.reshape([1, state.shape[0]]))[0]
        
def train(env, replay_memory, model, target_model, done):
    learning_rate = 0.7 # Learning rate
    discount_factor = 0.618
    MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return
    batch_size = 64 * 2
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([encode_observation(transition[0], env.observation_space.shape) for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([encode_observation(transition[3], env.observation_space.shape) for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        X.append(encode_observation(observation, env.observation_space.shape))
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)

def encode_observation(observation, n_dims):
    return observation

train_episodes = 100
num_actions = 3
actions = ['exploit','explore','random']
def main():
    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1 # You can't explore more than 100% of the time
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
    decay = 0.01
    # 1. Initialize the Target and Main models
    # Main Model (updated every 4 steps)
    env = environment(data)
    env.reset(100)
    main = agent(state_shape = env.current_state.shape, action_shape = num_actions,
                  learning_rate = 0.7)
    main_model = main.DQN()
    # Target Model (updated every 100 steps)
    target = agent(state_shape = env.current_state.shape, action_shape = num_actions,
                         learning_rate = 0.7)
    target_model = target.DQN()
    target_model.set_weights(main_model.get_weights())

    replay_memory = deque(maxlen=50_000)

    target_update_counter = 0

    # X = states, y = actions
    X = []
    y = []

    steps_to_update_target_model = 0

    for c,episode in enumerate(range(train_episodes)):
        random.seed(42+c*10)
        total_training_rewards = 0
        env.reset(100)
        observation = env.current_state
        done = False
        i = 0
        while not done:
            random_number = np.random.rand()
            print('number of training samples'+str(env.current_data.shape))
            i += 1
            if random_number <= epsilon:
                # Explore
                action = random.randint(0,2)
            # Exploit best known action (no epsilon sampling due to option of random step in actions)
            else: 
                encoded = encode_observation(observation, observation.shape[0])
                encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                predicted = main_model.predict(encoded_reshaped).flatten()
                action = np.argmax(predicted)
            new_observation, reward, done, info = env.step(action)
            replay_memory.append([observation, action, reward, new_observation, done])

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 4 == 0 or done:
                train(env, replay_memory, main_model, target_model, done)

            observation = new_observation
            total_training_rewards += reward
            env.update_model()
            print(env.m)
            print('distance to max')
            print(np.max(env.virtual_data[:,-1])-np.max(env.current_data[:,-1]))
            print('action = '+actions[action])     
            if done:
                print('Total training rewards: {} after n steps = {} with final reward = {}'.format(total_training_rewards, episode, reward))
                total_training_rewards += 1

                if steps_to_update_target_model >= 100:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(main_model.get_weights())
                    steps_to_update_target_model = 0
                break
            if i == 10000:
                break

if __name__ == '__main__':
    main()     
        