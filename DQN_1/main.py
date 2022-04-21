#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 10:56:39 2022

@author: timhartnett

"""
from environments import env_1
import numpy as np
import networks
import os
import random
import pandas as pd
from collections import deque

if __name__ == '__main__':
    random.seed(42)
    WORKING_DIR = '/Users/timhartnett/Downloads/RL_bayesian_optimization/'
    data_df = pd.read_csv(WORKING_DIR+'Dataset/train.csv')
    data_df = data_df.drop('number_of_elements',axis=1)
    data_df = data_df[['mean_atomic_mass','std_atomic_mass','mean_fie',
                  'mean_atomic_radius','std_atomic_radius',
                  'mean_ElectronAffinity','mean_FusionHeat',
                  'std_FusionHeat','mean_ThermalConductivity',
                  'std_ThermalConductivity','mean_Valence','entropy_Valence','std_Valence',
                  'critical_temp']]
    data_df = data_df.dropna()
    data = data_df.values
    print('... data loaded...')
    
    train_episodes = 1000
    actions = ['random walk','explore','exploit','ucb','pi','ei']
    num_actions = len(actions)
    
    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start# You can't explore more than 100% of the time
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
    decay = 0.001
    number_experiments = 100 # total number of steps allowed per episode

    env = env_1(data,n_initial=20)
    agent = networks.Agent(num_actions, epsilon = epsilon)
    score, eps_history = [], []
    
    for episode in range(train_episodes):
        random.seed(42+episode*10)
        total_training_rewards = 0
        env.reset()
        observation = env.current_state
        done = env.is_done()
        print('EPISODE ' + str(episode))
        for i in range(number_experiments):
            print(i)
            action = agent.choose_action(observation)
            new_observation, reward, done, info = env.step(action,i)
            total_training_rewards += reward
            agent.store_transition(observation, action, reward, new_observation, done)
            observation = new_observation
            agent.learn()
        eps_history.append(agent.epsilon)
        score.append(total_training_rewards)
    
    agent.save_models()
    
