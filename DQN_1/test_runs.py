#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 09:33:52 2022

Test runs

@author: timhartnett
"""
from environments import env_1
import numpy as np
import networks
import random
import pandas as pd

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

actions = ['random walk','explore','exploit','ucb','pi','ei']
num_actions = len(actions)
    
agent = networks.Agent(num_actions, epsilon = 0)
agent.load_models()


'''TEST RUNS '''
Test_runs = 100
number_experiments = 100
improvement_tracker = []
env = env_1(data,20)
for episode in range(Test_runs):
    print(str(episode))
    random.seed(42+episode*10)
    total_training_rewards = 0
    env.reset()
    observation = env.current_state
    done = False
    for i in range(number_experiments):
        encoded = encode_observation(observation, observation.shape[0])
        encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
        predicted = model_3.predict(encoded_reshaped).flatten()
        action = np.argmax(predicted)
        new_observation, reward, done, info = env.step(action,i)
        observation = new_observation
        total_training_rewards += reward
        env.update_model()
        
    improvement_tracker.append(np.max(env.current_data[:,-1])-env.initial_max)

improvement = np.array(improvement_tracker)
improvement = pd.DataFrame(improvement.astype('float'),columns = ['Increase_T_c']).to_csv('model_3_test.csv')
