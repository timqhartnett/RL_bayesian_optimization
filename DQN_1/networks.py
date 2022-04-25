#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:01:51 2022

Networks for DQN

@author: timhartnett
"""
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import random
from tensorflow.keras.models import load_model

class DQN(tf.keras.Model):
    
    def __init__(self, n_actions, fc1_dims = 64, fc2_dims=64, name = 'DQN',
                 checkpoint_dir = 'tmp/DQN/'):
        super(DQN,self).__init__()
        self.fc1_dims = fc1_dims  #number of dimensiion on first fully connected layer
        self.fc2_dims = fc2_dims # num dims second fully connected layer
        self.n_actions = n_actions # action space
        self.model_name = name #model name
        self.checkpoint_dir = checkpoint_dir #where to save model checkpoints
        self.checkpoint_file = os.path.join(self.checkpoint_dir+name+'_dqn_1') #file name

        self.fc1 = Dense(self.fc1_dims, activation = 'relu',input_shape=(self.n_actions,),
                         kernel_regularizer = tf.keras.regularizers.L2(0.01))
        self.fc2 = Dense(self.fc2_dims, activation = 'relu',
                         kernel_regularizer = tf.keras.regularizers.L2(0.01))
        self.q =  Dense(self.n_actions, activation='linear')
    
    def call(self,state):
        value = self.fc1(state)
        value = self.fc2(value)
        q = (self.q(value))
            
        return q

class ReplayBuffer():
    def __init__(self, max_size):
        self.replay_memory = deque(maxlen=max_size)
    
    def store_transition(self, state, action, reward, new_state, done):
        self.replay_memory.append([state, action, reward, new_state, done])
    
    def sample_buffer(self,batch_size):
        mini_batch = random.sample(self.replay_memory, batch_size)
        states = np.array([item[0] for item in mini_batch])
        actions = np.array([item[1] for item in mini_batch])
        rewards = np.array([item[2] for item in mini_batch])
        new_states = np.array([item[3] for item in mini_batch])
        dones = np.array([item[4] for item in mini_batch])
    
        return states, actions, rewards, new_states, dones

class Agent:
    def __init__(self,n_actions,epsilon,epsilon_dec = 1e-3, eps_end = 0.01, alpha=0.01,gamma=0.99,
                 batch_size=128,mem_size = 1000000, fc1_dims = 64, fc2_dims = 64,
                 replace = 100,fname = 'DQN_1'):
        self.action_space = [i for i in range(n_actions)]
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_end = eps_end
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.replace = replace
        self.model_file = '/Users/timhartnett/Downloads/RL_bayesian_optimization/DQN_1/'+fname
        
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size)
        self.q_eval = DQN(n_actions = n_actions)
        self.target = DQN(n_actions = n_actions)

        
        self.q_eval.compile(optimizer=Adam(learning_rate=alpha),loss='mean_squared_error')
        self.target.compile(optimizer=Adam(learning_rate=alpha),loss='mean_squared_error')

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval(state)
            action = tf.math.argmax(actions,axis=1).numpy()[0]
        
        return action
    
    def learn(self):
        if len(self.memory.replay_memory) < self.batch_size:
            return
        
        if self.learn_step_counter % self.replace == 0:
            self.target.set_weights(self.q_eval.get_weights())
        
        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)
        
        q_pred = self.q_eval(states)
        q_next = tf.math.reduce_max(self.target(new_states),axis=1, keepdims=True).numpy()
        q_target = np.copy(q_pred)
        
        for idx, terminal in enumerate(dones):
            if terminal:
                q_next[idx] = 0.0
            q_target[idx,actions[idx]] = rewards[idx] + self.gamma*q_next[idx]
        
        self.q_eval.train_on_batch(states,q_target)
        
        if self.epsilon > self.eps_end:
            self.epsilon = self.epsilon - self.eps_dec
        
        self.learn_step_counter += 1
        
    def save_model(self):
        self.q_eval.save(self.model_file)
        
    def load_model(self):
        self.q_eval = load_model(self.model_file)
        
        
            
        
    
        
        
    
    
        
        
        
        