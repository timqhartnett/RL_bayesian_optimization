#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 10:33:37 2021

Expected Improvement

@author: timothy
"""
'''
  minimum = min(data$X66)
  z = (minimum-me$mean)/me$sd
  ei = ((minimum-me$mean)*(pnorm(z)))+(me$sd*dnorm(z))
  me$EI = ei
  me_virtual= cbind(data.virtual, me)
  ordering_run1 = me_virtual[order(me_virtual$EI, decreasing=TRUE),]
  print("EXPECTED IMPROVEMENT RANKING")
  print(head(ordering_run1))
  print("END EXPECTED IMPROVEMENT RANKING")
  ei_track = as.data.frame(max(ei))
'''
import pandas as pd #stable version 1.3.0
import os
import gpflow #stable version 2.2.1
import numpy as np #stable version 1.19.2
from tensorflow import keras #stable version 2.5.0
import tensorflow as tf #stable version 2.5.0
from gpflow.utilities import print_summary
import random
from collections import deque
from scipy.stats import norm


random.seed(42)
WORKING_DIR = os.getcwd()
data_df = pd.read_csv(WORKING_DIR+'/train.csv')

data_df = data_df.drop('number_of_elements',axis=1)
data_df = data_df[['mean_atomic_mass','std_atomic_mass','mean_fie',
                  'mean_atomic_radius','std_atomic_radius',
                  'mean_ElectronAffinity','mean_FusionHeat',
                  'std_FusionHeat','mean_ThermalConductivity',
                  'std_ThermalConductivity','mean_Valence','entropy_Valence','std_Valence',
                  'critical_temp']]

correlation = data_df.corr()
#data_df = data_df.drop(['mean_Density','wtd_mean_Density','gmean_Density','wtd_mean_Density','std_Density',
#                        'wtd_std_Density','range_Density','wtd_range_Density'],axis=1)
data = data_df.values
class GP_model(object):
    '''
    model for bayesian prediction of material properties
    '''
    def __init__(self,total_data):
        self.initial_data = total_data
    
    def train_GP(self,data,kernel='Matern32',optimizer='L-BFGS-B'):
        X = data[:,:-1]
        Y = data[:,-1].reshape(-1,1)
        X = (X - X.mean()) / X.std()
        Y = (Y - Y.mean()) / Y.std()
        k = gpflow.kernels.Matern52()
        m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)
        opt = gpflow.optimizers.Scipy()
        self.opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
        return m
    
    def train_initial_model(self,number_initial_samples):
        self.virtual_data = self.initial_data.copy()
        initial_selection = np.unique(random.sample(range(self.virtual_data.shape[0]),number_initial_samples))
        self.training_data = self.virtual_data[initial_selection,:]
        self.virtual_data = np.delete(self.virtual_data,initial_selection,axis=0)
        self.m = self.train_GP(self.training_data)
        self.initial_max = np.max(self.training_data[:,-1])
    
    def predict_virtual(self):
        X = self.virtual_data[:,:-1]
        X = (X - X.mean()) / X.std()
        norm_mean, norm_var = self.m.predict_y(X)
        mean = norm_mean*self.training_data[:,-1].std()+self.training_data[:,-1].mean()
        var = norm_var*self.training_data[:,-1].std()+self.training_data[:,-1].mean()
        return mean, var
    
    def Expected_Improvement(self):
        dist = norm
        mean, var = self.predict_virtual()
        maximum = np.max(self.training_data[:,-1])
        z = (maximum-mean)/var
        ei = (maximum+mean)*dist.cdf(z)+(var*dist.pdf(z))
        return ei
    
    def update_model(self):
        ei = self.Expected_Improvement()
        new_data_index = np.argmax(ei)
        new_data = self.virtual_data[new_data_index,:]
        self.training_data = np.unique(np.vstack((self.training_data,new_data)),axis=0)
        self.m = self.train_GP(self.training_data)
        

'''TEST RUNS '''
Test_runs = 100
improvement_tracker = []

for episode in range(Test_runs):
    print(str(episode))
    random.seed(42+episode*10)
    number_experiments = random.randint(10,100)
    total_training_rewards = 0
    model = GP_model(data)
    model.train_initial_model(random.randint(20,100))
    
    for i in range(number_experiments):
        model.update_model()
        
    improvement_tracker.append(np.max(model.training_data[:,-1])-model.initial_max)

improvement = np.array(improvement_tracker)
improvement = pd.DataFrame(improvement.astype('float'),columns = ['Increase_T_c']).to_csv('ei_test.csv')

        
        






