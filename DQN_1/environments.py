
import pandas as pd #stable version 1.3.0
import gpflow #stable version 2.2.1
import numpy as np #stable version 1.19.2
from gpflow.utilities import print_summary
from bayes_util import GP_model
from bayes_util import aquisition_functions as aqf
import random


class env_1(object):
    '''
    Environment is described by normalized values of training value (y-mu)/sd.
    reward is increase in Tc at each step plus 200 if Tc max is found
    '''
    
    def __init__(self,virtual_data,n_initial):
        self.initial_data = virtual_data
        self.virtual_data = self.initial_data.copy()
        self.n_initial = n_initial
        initial_selection = np.unique(random.sample(range(self.virtual_data.shape[0]),n_initial))
        self.current_data = self.virtual_data[initial_selection,:]
        self.virtual_data = np.delete(self.virtual_data,initial_selection,axis=0)
        model = GP_model(self.current_data)
        self.m = model.train_GP()
        self.current_state = self.state(self.current_data,0)
    
    def update_model(self):
        model = GP_model(self.current_data)
        m_ = model.train_GP()
        if m_ is not None:
            self.m = m_
    
    def predict_virtual(self):
        X = self.virtual_data[:,:-1]
        X = (X - X.mean()) / X.std()
        norm_mean, norm_var = self.m.predict_y(X)
        mean = norm_mean*self.current_data[:,-1].std()+self.current_data[:,-1].mean()
        var = norm_var*self.current_data[:,-1].std()+self.current_data[:,-1].mean()
        return mean, var
    
    def state(self,current_data,exp_num):
        mean = np.mean(current_data[:,-1])
        sd = np.std(current_data[:,-1])
        norm_data = (current_data[:,-1]-mean)/sd
        norm_data.sort()
        virtual_mean,virtual_var = self.predict_virtual()
        max_diff = np.max(virtual_mean)-np.max(current_data[:,-1])
        virtual_var = np.mean(virtual_var)
        state = np.array([norm_data[0],norm_data[-1],max_diff,virtual_var,exp_num])
        return state
    
    def reset(self):
        '''
        this method is to generate a random initial training set for a toy problem and train the initial GP
        '''
        self.virtual_data = self.initial_data.copy()
        initial_selection = np.unique(random.sample(range(self.virtual_data.shape[0]),self.n_initial))
        self.current_data = self.virtual_data[initial_selection,:]
        self.virtual_data = np.delete(self.virtual_data,initial_selection,axis=0)
        model = GP_model(self.current_data)
        self.m = model.train_GP()
        self.current_state = self.state(self.current_data,0)
    
    def step(self,action,experiment_number):
        mean, var = self.predict_virtual()
        aq_func = aqf(mean,var,self.current_data[:,-1],experiment_number)
        if action == 0:
            info = 'random walk'
            new_data_index = aq_func.random_walk()
        elif action == 1:
            info = 'explore'
            new_data_index = aq_func.explore()
        elif action == 2:
            info = 'exploit'
            new_data_index = aq_func.exploit()
        elif action == 3:
            info = 'ucb'
            new_data_index = aq_func.ucb()
        elif action == 4:
            info = 'pi'
            new_data_index = aq_func.pi()
        elif action == 5:
            info = 'ei'
            new_data_index = aq_func.ei()
            
        new_data = self.virtual_data[new_data_index,:]
        reward = self.virtual_data[new_data_index,-1]- np.max(self.current_data[:,-1])
        
        if reward < 0:
            reward = 0
        
        self.current_data = np.unique(np.vstack((self.current_data,new_data)),axis=0)

        new_state = self.state(self.current_data,experiment_number)
        self.current_state = new_state
        self.update_model()
        
        done = self.is_done()
        if done:
            reward = reward+200
        
        self.virtual_data = np.delete(self.virtual_data,new_data_index,axis=0)
        
        return new_state, reward, done, info
    
    def is_done(self):
        if np.max(self.virtual_data[:,-1]) <= np.max(self.current_data[:,-1]):
            done = True 
        else:
            done = False
        return done
    
