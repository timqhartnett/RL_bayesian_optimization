# Bayesian Optimization Utilities
import gpflow
import numpy as np
from scipy.stats import norm
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

class GP_model(object):
    '''
    model for bayesian prediction of material properties
    '''
    def __init__(self,training_data):
        self.training_data = training_data

    def train_GP(self,kernel='Matern32',optimizer='L-BFGS-B'):
        X = self.training_data[:,:-1]
        Y = self.training_data[:,-1].reshape(-1,1)
        X = (X - X.mean()) / X.std()
        Y = (Y - Y.mean()) / Y.std()
        k = gpflow.kernels.Matern52()
        m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)
        opt = gpflow.optimizers.Scipy()
        try:
            self.opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
            return m
        except:
            return None

class mc_ensemble_nn(tf.keras.Model):
    def __init__(self, input_cols,dropout_rate = 0.1, alpha = 0.001,fc1_dims= 32,fc2_dims=32,
    name = 'NN_ensemble',checkpoint_dir = 'tmp/actor_critic/'):
        super(mc_ensemble_nn,self).__init__()
        self.model = Sequential([
            Dense(128,  activation="relu", input_shape=(input_cols,)),
            Dropout(dropout_rate),
            Dense(64, activation="relu"),
            Dropout(dropout_rate),
            Dense(32, activation="relu"),
            Dropout(dropout_rate),
            Dense(1)
    ])
        self.model.compile(optimizer=Adam(learning_rate = alpha),loss='mse')
        
    def train(self,X_train,Y_train):
        self.history = self.model.fit(X_train,Y_train,epochs = 100, validation_split=0.1)
    
    def predict_dist(self, X, model, num_samples):
        preds = [self.model(X, training=True) for _ in range(num_samples)]
        return np.hstack(preds)
    
    def predict_mean_sd(self, X, num_samples=100):
        pred_dist = self.predict_dist(X, self.model, num_samples)
        
        return pred_dist.mean(axis=1), pred_dist.std(axis=1)

        
class aquisition_functions():
    def __init__(self, virtual_means, virtual_vars, training_data, step_num):
        self.means = virtual_means
        self.vars = virtual_vars
        self.train = training_data
        self.step_num = step_num

    def random_walk(self):
        new_data_index = np.random.choice(range(len(self.means)),1)
        return new_data_index
    
    def explore(self):
        new_data_index = np.argmax(self.vars)
        return new_data_index

    def exploit(self):
        new_data_index = np.argmax(self.means)
        return new_data_index
    
    def ucb(self,alpha = 0.5,decay_rate=1):
        kappa = 1/(1+self.step_num)*alpha
        ucb = (self.means - max(self.train)) + kappa * self.vars
        new_data_index = np.argmax(ucb)
        return new_data_index
    
    def pi(self,alpha = 0.5,decay_rate=1):
        kappa = 1/(1+decay_rate*self.step_num)*alpha
        gamma_x = (self.means - (max(self.train) + kappa)) / self.vars
        pi = norm.cdf(gamma_x)
        new_data_index = np.argmax(pi)
        return new_data_index

    def ei(self):
        dist = norm
        maximum = np.max(self.train)
        z = (maximum-self.means)/self.vars
        ei = (maximum-self.means)*dist.cdf(z)+(self.vars*dist.pdf(z))
        new_data_index = np.argmax(ei)
        return new_data_index